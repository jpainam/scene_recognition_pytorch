import argparse
import os.path as osp
import sys
import time

import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torchsummary import summary
import models
from data.data_manager import get_data
from evaluation.evaluation import Evaluation
from trainer import ClassificationTrainer
from utils.loggers import Logger, mkdir_if_missing


parser = argparse.ArgumentParser(description='Scene Recognition Training Procedure')
parser.add_argument('--config', metavar='DIR', help='Configuration file path')
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.config, 'r'))

reweighting = CONFIG['MODEL']['ARM']
with_attribute = CONFIG['MODEL']['WITH_ATTRIBUTE']
backbone_name = CONFIG['MODEL']['BACKBONE']
ext_f = "baseline"
ext_f = "32x16d"
if reweighting:
    ext_f = "arm"
elif with_attribute:
    ext_f = "attribute"
output_folder = "{}{}_{}".format(backbone_name,
                                 CONFIG['MODEL']['ARCH'],
                                 ext_f)

save_dir = osp.join('./logs', CONFIG['DATASET']['NAME'], output_folder)
checkpoint = osp.join('./checkpoints',
                      CONFIG['DATASET']['NAME'], output_folder)

mkdir_if_missing(save_dir)
mkdir_if_missing(checkpoint)
log_name = f"train_{time.strftime('-%Y-%m-%d-%H-%M-%S')}.log"
sys.stdout = Logger(osp.join(save_dir, log_name))

timestamp = time.strftime("0:%Y-%m-%dT%H-%M-%S")
summary_writer = SummaryWriter(osp.join(save_dir, 'tensorboard_log' + timestamp))

if __name__ == "__main__":

    train_loader, val_loader, class_names, attrs = get_data(dataset=CONFIG['DATASET']['NAME'],
                                                            root=CONFIG['DATASET']['ROOT'],
                                                            train_folder=CONFIG['DATASET']['TRAIN'],
                                                            val_folder=CONFIG['DATASET']['VAL'],
                                                            ten_crops=CONFIG['TESTING']['TEN_CROPS'],
                                                            batch_size=CONFIG['TRAINING']['BATCH_SIZE'],
                                                            with_attribute=with_attribute)

    print('checkpoint dir {}'.format(checkpoint))
    print('save_dir/logs dir {}'.format(save_dir))
    model = models.get_model(num_classes=len(class_names),
                             with_attribute=with_attribute,
                             with_reweighting=with_attribute,
                             num_attrs=len(attrs),
                             backbone=CONFIG['MODEL']['BACKBONE'],
                             arch=CONFIG['MODEL']['ARCH'])

    ignored_params = list(map(id, model.features.parameters()))
    classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    params = [{"params": model.features.parameters(), "lr": 0.01},
                {"params": classifier_params, "lr": 0.1}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer = torch.optim.Adam(params, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    use_cuda = torch.cuda.is_available()

    if with_attribute:
        criterion = [nn.CrossEntropyLoss(), nn.BCELoss()]
    else:
        criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    summary(model, (3, 224, 224))
    trainer = ClassificationTrainer(model=model, train_loader=train_loader,
                                    val_loader=val_loader,
                                    num_attrs=len(attrs),
                                    save_dir=save_dir,
                                    attribute_list=attrs,
                                    with_attribute=with_attribute,
                                    summary_writer=summary_writer,
                                    num_classes=CONFIG['DATASET']['NUM_CATEGORY'],
                                    criterion=criterion, optimizer=optimizer, config=CONFIG)

    evaluate = Evaluation(model=model,
                          dataloader=val_loader,
                          classes=class_names,
                          ten_crops=CONFIG['TESTING']['TEN_CROPS'],
                          with_attribute=with_attribute)

    epochs = CONFIG['TRAINING']['EPOCH']
    for epoch in range(epochs):
        print("Epoch {}".format(epoch + 1))
        epoch_time = time.time()
        trainer.train(epoch)
        scheduler.step()
        print("LR1/LR2: [{}/{}], Train Time: {:.2f}".format(
            optimizer.param_groups[0]['lr'],
            optimizer.param_groups[1]['lr'],
            time.time() - epoch_time
        ))

        print('-' * 60)

        evaluate.test(topk=(1, 2, 5))
        # trainer.eval(epoch)

        if (epoch + 1) % 10 == 0:
            try:
                torch.save(model.module.state_dict(), f"{checkpoint}/model_{epoch + 1}.pth.tar")
            except AttributeError:
                torch.save(model.state_dict(), f"{checkpoint}/model_{epoch + 1}.pth.tar")
        print('-' * 60)
