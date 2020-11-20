import torch
from models.resnet import *
from torch import nn
import sys
import os.path as osp
from utils.loggers import Logger
import time
from evaluation.classification import accuracy
from utils.meters import AverageMeter
from data.data_manager import get_data
import argparse
import yaml

parser = argparse.ArgumentParser(description='Scene Recognition Training Procedure')
parser.add_argument('--config', metavar='DIR', help='Configuration file path')
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.config, 'r'))

save_dir = osp.join(CONFIG['MODEL']['OUTPUT'], CONFIG['DATASET']['NAME'])
log_name = f"train_{time.strftime('-%Y-%m-%d-%H-%M-%S')}.log"
sys.stdout = Logger(osp.join(save_dir, log_name))


if __name__ == "__main__":

    train_loader, val_loader, class_names = get_data(root=CONFIG['DATASET']['ROOT'])
    model = resnet50(pretrained=True, num_classes=len(class_names), num_features=512)

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([{"params": base_params, "lr": 0.001},
                                 {"params": model.classifier.parameters(), "lr": 0.01}],
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    use_cuda = torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()

    epochs = 50
    dataloader = {"train": train_loader, "val": val_loader}
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - epoch))
        print("-" * 65)
        for phase in ['train', 'val']:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            prec1 = AverageMeter()
            prec2 = AverageMeter()
            prec5 = AverageMeter()
            print(f'Running {phase} phase')
            # training
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            end = time.time()
            for i, (images, labels) in enumerate(dataloader[phase]):
                data_time.update(time.time() - end)
                if use_cuda:
                    images, labels = images.cuda(), labels.cuda()

                if phase == "train":
                    outputs = model(images)
                else:
                    with torch.no_grad():
                        outputs = model(images)

                loss = criterion(outputs, labels)
                prec = accuracy(outputs.data, labels.data, topk=(1, 2, 5))

                losses.update(loss.item(), labels.size(0))
                prec1.update(prec[0].item(), labels.size(0))
                prec2.update(prec[1].item(), labels.size(0))
                prec5.update(prec[2].item(), labels.size(0))

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % 10 == 0:
                    print('Epoch: [{}][{}]\t'
                          'Loss {:.3f} ({:.3f})\t'
                          'Prec1 {:.2%} ({:.2%})\t'
                          'Prec2 {:.2%} ({:.2%})\t'
                          'Prec5 {:.2%} ({:.2%})\t'
                          .format(epoch, i + 1,
                                  losses.val, losses.avg,
                                  prec1.val, prec2.avg,
                                  prec2.val, prec2.avg,
                                  prec5.val, prec5.avg))

            print('Summary for {}'.format(phase))
            print('Epoch: [{}][{}]\t'
                  'Loss {:.3f} ({:.3f})\t'
                  'Prec1 {:.2%} ({:.2%})\t'
                  'Prec2 {:.2%} ({:.2%})\t'
                  'Prec5 {:.2%} ({:.2%})\t'
                  .format(epoch, i + 1,
                          # batch_time.val, batch_time.avg,
                          # data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          prec1.val, prec2.avg,
                          prec2.val, prec2.avg,
                          prec5.val, prec5.avg))
            print()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./logs/model_{epoch + 1}.pth.tar')

        print()
