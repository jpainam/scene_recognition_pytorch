import time
from utils.meters import AverageMeter
from evaluation.classification import accuracy, accuracy_multilabel2, precision
import torch
from torch.nn.utils import  clip_grad_norm_


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer,
                 config,
                 summary_writer,
                 num_classes,
                 xi=.8,
                 with_attribute=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.summary_writer = summary_writer
        self.with_attribute = with_attribute
        assert num_classes != 0
        self.num_classes = num_classes
        self.xi = xi

    def train(self, epoch):
        raise NotImplementedError

    def eval(self, epoch):
        raise NotImplementedError


class AttributeTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        correct = AverageMeter()
        losses = AverageMeter()
        prec1 = AverageMeter()
        prec2 = AverageMeter()
        prec5 = AverageMeter()

        for step, (imgs, labels, orig_attrs) in enumerate(self.train_loader):

            imgs, labels = imgs.cuda(), labels.cuda()
            pred_attrs = []
            if self.with_attribute:
                orig_attrs = orig_attrs.cuda()
                attrs = orig_attrs.detach().clone()
                attrs[attrs > self.xi] = 1.
                attrs[attrs <= self.xi] = 0.
                pred_id, pred_attrs = self.model(imgs, orig_attrs)
                assert pred_attrs.shape[-1] == 134
            else:
                pred_id = self.model(imgs, orig_attrs)
            assert pred_id.shape[-1] == self.num_classes

            if self.with_attribute:
                loss = self.criterion[0](pred_id, labels)
                loss_attrs = self.criterion[1](pred_attrs.float(), attrs.float())
                if epoch > 15:
                    loss += loss_attrs
            else:
                loss = self.criterion(pred_id, labels)

            #clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prec = accuracy(pred_id.data, labels.data, topk=(1, 2, 5))

            losses.update(loss.item(), labels.size(0))
            prec1.update(prec[0].item(), labels.size(0))
            prec2.update(prec[1].item(), labels.size(0))
            prec5.update(prec[2].item(), labels.size(0))
            y_pred = pred_id.argmax(dim=1)
            acc = (y_pred == labels).sum().item() / labels.size(0) * 100
            correct.update(acc, labels.size(0)/100.)

            # tensorboard
            if self.summary_writer is not None:
                global_step = epoch * len(self.train_loader) + step
                self.summary_writer.add_scalar('train_loss', loss.item(), global_step)
                self.summary_writer.add_scalar('train_acc', 1. * correct.avg, global_step)
                self.summary_writer.add_scalar('prec1', prec1.avg, global_step)
                self.summary_writer.add_scalar('prec2', prec2.avg, global_step)
                self.summary_writer.add_scalar('prec5', prec5.avg, global_step)

            if (step + 1) % 10 == 0:
                print('[{}] '
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
            for step, (imgs, labels, orig_attrs) in enumerate(self.val_loader):
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.with_attribute:
                    orig_attrs = orig_attrs.cuda()
                    attrs = orig_attrs.detach().clone()
                    attrs[attrs > self.xi] = 1.
                    attrs[attrs <= self.xi] = 0.
                    pred_id, pred_attrs = self.model(imgs, orig_attrs)
                    assert pred_attrs.shape[-1] == 134
                    loss = self.criterion[0](pred_id, labels)
                    loss_attrs = self.criterion[1](pred_attrs.float(), attrs.float())
                    if epoch > 15:
                        loss += loss_attrs
                else:
                    pred_id = self.model(imgs, orig_attrs)
                    loss = self.criterion(pred_id, labels)
                assert pred_id.shape[-1] == self.num_classes
                losses.update(loss.item(), labels.size(0))

                prec = accuracy(pred_id.data, labels.data, topk=(1, 2, 5),
                                is_multilabel=False)
                prec1.update(prec[0].item(), labels.size(0))
                prec2.update(prec[1].item(), labels.size(0))
                prec5.update(prec[2].item(), labels.size(0))

                y_pred = pred_id.argmax(dim=1)
                acc = (y_pred == labels).sum().item() / labels.size(0) * 100
                correct.update(acc, labels.size(0)/100.)

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
