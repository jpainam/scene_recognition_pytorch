import torch
from models.resnet import *
from torch import nn
import sys
import os.path as osp
from utils.loggers import Logger, mkdir_if_missing
import time
from evaluation.classification import accuracy, accuracy_multilabel2, accuracy_multilabel
from utils.meters import AverageMeter
from data.data_manager import get_data
import argparse
import yaml
from tensorboardX import SummaryWriter
import models
from trainer import AttributeTrainer
from loss.attribute_loss import CEL_Sigmoid
import numpy as np
from evaluation.classification import get_attribute_results
from utils import time_str

parser = argparse.ArgumentParser(description='Scene Recognition Training Procedure')
parser.add_argument('--config', metavar='DIR', help='Configuration file path')
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.config, 'r'))

save_dir = osp.join(CONFIG['TRAINING']['LOG_DIR'], CONFIG['DATASET']['NAME'])
mkdir_if_missing(save_dir)
mkdir_if_missing(CONFIG['MODEL']['CHECKPOINTS'])
log_name = f"train_{time.strftime('-%Y-%m-%d-%H-%M-%S')}.log"
sys.stdout = Logger(osp.join(save_dir, log_name))

timestamp = time.strftime("0:%Y-%m-%dT%H-%M-%S")
summary_writer = SummaryWriter(osp.join(save_dir, 'tensorboard_log' + timestamp))

with_attribute = CONFIG['TRAINING']['WITH_ATTRIBUTE']

if __name__ == "__main__":

    train_loader, val_loader, class_names, attrs = get_data(dataset=CONFIG['DATASET']['NAME'],
                                                            root=CONFIG['DATASET']['ROOT'],
                                                            ten_crops=CONFIG['TESTING']['TEN_CROPS'],
                                                            batch_size=CONFIG['TRAINING']['BATCH_SIZE'],
                                                            with_attribute=CONFIG['TRAINING']['WITH_ATTRIBUTE'])
    model = models.get_model(CONFIG,
                             num_classes=len(class_names),
                             num_attrs=len(attrs) if with_attribute else 0,
                             dropout=0.5)

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([{"params": base_params, "lr": 0.001},
                                 {"params": model.classifier.parameters(), "lr": 0.01}],
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    use_cuda = torch.cuda.is_available()
    # criterion = [nn.CrossEntropyLoss(), nn.MultiLabelSoftMarginLoss()]
    # criterion = CEL_Sigmoid(attrs.mean(0))
    # criterion = nn.MultiLabelMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer = AttributeTrainer(model=model, train_loader=train_loader, val_loader=val_loader,
                               summary_writer=summary_writer, criterion=criterion, optimizer=optimizer, config=CONFIG)

    epochs = CONFIG['TRAINING']['EPOCH']
    dataloader = {"train": train_loader, "val": val_loader}
    for epoch in range(epochs):
        train_loss, train_gt, train_probs = trainer.train(epoch)

        valid_loss, valid_gt, valid_probs = trainer.eval(epoch)
        train_result = get_attribute_results(train_gt, train_probs)

        valid_result = get_attribute_results(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)
