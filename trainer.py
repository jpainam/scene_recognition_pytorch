import time
from utils.meters import AverageMeter
from evaluation.classification import accuracy, accuracy_multilabel2, precision
import torch
import numpy as np
from tqdm import tqdm
from utils import time_str
from sklearn.metrics import precision_score

class BaseTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, config,
                 summary_writer,
                 val_loader,
                 with_attribute=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.summary_writer = summary_writer
        self.with_attribute = with_attribute

    def train(self, epoch):
        raise NotImplementedError

    def eval(self, epoch):
        raise NotImplementedError


class ClassificationTrainer(BaseTrainer):
    def __init__(self):
        super(ClassificationTrainer, self).__init__()

    '''def batch_trainer(self, epoch):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 65)
        for phase in ['train']:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            prec1 = AverageMeter()
            prec2 = AverageMeter()
            prec5 = AverageMeter()
            print(f'Running {phase} phase')
            # training
            if phase == "train":
                self.model.train(True)
            else:
                model.train(False)
            end = time.time()
            for i, (images, labels) in enumerate(dataloader[phase]):
                data_time.update(time.time() - end)
                images, labels = images.cuda(), labels.cuda()
                    # attributes = attributes.cuda() if with_attribute else None

                if phase == "val" and CONFIG['TESTING']['TEN_CROPS']:
                    bs, ncrops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                if phase == "train":
                    outputs, feat, attrs = self.model(images)
                else:
                    with torch.no_grad():
                        outputs, feat, attrs = self.model(images)

                if phase == "val" and CONFIG['TESTING']['TEN_CROPS']:
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                loss = self.criterion[1](outputs, labels)
                # if with_attribute:
                #    loss += criterion[1](attrs, attributes)

                # prec = accuracy(outputs.data, labels.data, topk=(1, 2, 5))
                prec = accuracy_multilabel(outputs.data, labels.data)

                losses.update(loss.item(), labels.size(0))
                prec1.update(prec.item(), labels.size(0))
                # prec1.update(prec[0].item(), labels.size(0))
                # prec2.update(prec[1].item(), labels.size(0))
                # prec5.update(prec[2].item(), labels.size(0))

                # tensorboard
                if summary_writer is not None:
                    global_step = epoch * len(dataloader[phase]) + i
                    summary_writer.add_scalar(f'{phase}_loss', loss.item(), global_step)
                    summary_writer.add_scalar(f'{phase}_prec1', prec.item(), global_step)
                    # summary_writer.add_scalar(f'{phase}_prec2', prec[1].item(), global_step)
                    # summary_writer.add_scalar(f'{phase}_prec5', prec[2].item(), global_step)

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
                          prec1.val, prec1.avg,
                          prec2.val, prec2.avg,
                          prec5.val, prec5.avg))
            print()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            checkpoint = osp.join(CONFIG['MODEL']['CHECKPOINTS'], CONFIG['DATASET']['NAME'])
            try:
                torch.save(model.module.state_dict(), f"{checkpoint}/model_{epoch + 1}.pth.tar")
            except AttributeError:
                torch.save(model.state_dict(), f"{checkpoint}/model_{epoch + 1}.pth.tar")
                
    '''


class AttributeTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        correct = AverageMeter()
        losses = AverageMeter()
        prec1 = AverageMeter()
        prec2 = AverageMeter()
        prec5 = AverageMeter()

        for step, (imgs, labels, attrs) in enumerate(self.train_loader):

            imgs, labels = imgs.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            if self.with_attribute:
                attrs = attrs.cuda(),
                attrs = attrs.float()
                pred_id, pred_attrs = self.model(imgs)
                assert pred_attrs.shape[-1] == 134
            else:
                pred_id = self.model(imgs)
            assert pred_id.shape[-1] == self.config['DATASET']['NUM_CATEGORY']

            if self.with_attribute:
                loss_id = self.criterion[0](pred_id, labels)
                loss_attrs = self.criterion[1](pred_attrs, attrs)
                loss = loss_id + loss_attrs
            else:
                loss = self.criterion(pred_id, labels)

            loss.backward()
            self.optimizer.step()
            prec = accuracy(pred_id.data, labels.data, topk=(1, 2, 5))

            losses.update(loss.item(), labels.size(0))
            prec1.update(prec[0].item(), labels.size(0))
            prec2.update(prec[1].item(), labels.size(0))
            prec5.update(prec[2].item(), labels.size(0))
            _, predicted = torch.max(pred_id.data, dim=1)
            acc = (predicted == labels).sum().item()
            #number_of_correct = torch.sum(preds == attrs.bool()).item()
            #total_correct = attrs.size(0) * attrs.size(1)
            correct.update(acc, labels.size(0))

            # tensorboard
            if self.summary_writer is not None:
                global_step = epoch * len(self.train_loader) + step
                self.summary_writer.add_scalar('train_loss', loss.item(), global_step)
                self.summary_writer.add_scalar('train_acc', 1. * correct.avg, global_step)
                self.summary_writer.add_scalar('prec1', prec1.avg, global_step)
                self.summary_writer.add_scalar('prec2', prec2.avg, global_step)
                self.summary_writer.add_scalar('prec5', prec5.avg, global_step)

            if (step + 1) % 10 == 0:
                print('Train: [{}] '
                      'Loss {:.3f} ({:.3f})\t'
                      'Acc {:.2f} ({:.2f})\t'
                      'Prec1 {:.2%} ({:.2%})\t'
                      'Prec2 {:.2%} ({:.2%})\t'
                      'Prec5 {:.2%} ({:.2%})\t'
                      .format(step + 1,
                              losses.val, losses.avg,
                              correct.val, correct.avg,
                              prec1.val, prec2.avg,
                              prec2.val, prec2.avg,
                              prec5.val, prec5.avg
                              ))

        return correct.avg, losses.avg

    def eval(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        correct = AverageMeter()
        prec1 = AverageMeter()
        prec2 = AverageMeter()
        prec5 = AverageMeter()
        with torch.no_grad():
            for step, (imgs, labels, attrs) in enumerate(self.val_loader):
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.with_attribute:
                    attrs = attrs.cuda()
                    attrs = attrs.float()
                    pred_id, pred_attrs = self.model(imgs)
                    assert pred_attrs.shape[-1] == 134
                    loss_id = self.criterion[0](pred_id, labels)
                    loss_attrs = self.criterion[1](pred_attrs, attrs)
                    loss = loss_id + loss_attrs
                else:
                    pred_id = self.model(imgs)
                    loss = self.criterion(pred_id, labels)
                assert pred_id.shape[-1] == self.config['DATASET']['NUM_CATEGORY']
                losses.update(loss.item(), labels.size(0))

                prec = accuracy(pred_id.data, labels.data, topk=(1, 2, 5))
                prec1.update(prec[0].item(), labels.size(0))
                prec2.update(prec[1].item(), labels.size(0))
                prec5.update(prec[2].item(), labels.size(0))

                _, predicted = torch.max(pred_id.data, dim=1)
                acc = (predicted == labels).sum().item()
                correct.update(acc, labels.size(0))

        print('Val: [{}] '
              'Loss {:.2f} ({:.2f})\t'
              'Acc {:.2f} ({:.2f})\t'
              'Prec1 {:.2%} ({:.2%})\t'
              'Prec2 {:.2%} ({:.2%})\t'
              'Prec5 {:.2%} ({:.2%})\t'
              .format(epoch,
                      losses.val, losses.avg,
                      correct.val, correct.avg,
                      prec1.val, prec1.avg,
                      prec2.val, prec2.avg,
                      prec5.val, prec5.avg
                      ))

        return correct.avg, losses.avg
