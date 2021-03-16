import time
from utils.meters import AverageMeter
from evaluation.classification import accuracy, accuracy_multilabel2, precision
import torch
from torch.nn.utils import  clip_grad_norm_
import scipy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base_trainer import BaseTrainer
import os
import json


def check_metric_valid(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True


class AttributeTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        correct = AverageMeter()
        losses = AverageMeter()
        running_loss = 0.0
        running_corrects = 0

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
                    loss += loss_attrs * 134. / pred_attrs.size(0)
            else:
                loss = self.criterion(pred_id, labels)

            loss = self.criterion[1](pred_attrs.float(), attrs.float())

            self.optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            preds = torch.gt(pred_attrs, torch.ones_like(attrs) / 2)
            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == attrs.byte()).item() / attrs.size(0)
            predicted = pred_id.argmax(dim=1)
            acc = (predicted == labels).sum().item()
            correct.update(acc, labels.size(0))

            if (step + 1) % 10 == 0:
                print('step: ({}/{})  |  label loss: {:.4f}'.format(
                    step * labels.size(0), len(self.train_loader), loss.item()))

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_corrects / len(self.train_loader)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
        return correct.avg, losses.avg

    def eval(self, epoch):
        self.model.eval()
        preds_tensor = np.empty(shape=[0, self.num_attrs], dtype=np.byte)  # shape = (num_sample, num_label)
        attrs_tensor = np.empty(shape=[0, self.num_attrs], dtype=np.byte)  # shape = (num_sample, num_label)

        # Iterate over data.
        with torch.no_grad():
            for step, (images, labels, orig_attrs) in enumerate(self.val_loader):
                images, labels = images.cuda(), labels.cuda()
                orig_attrs = orig_attrs.cuda()
                attrs = orig_attrs.detach().clone()
                attrs[attrs <= self.xi] = 0.
                attrs[attrs > self.xi] = 1.0
                pred_id, pred_attr = self.model(images, orig_attrs)
                preds = torch.gt(pred_attr, torch.ones_like(pred_attr) / 2)
                # transform to numpy format
                attrs = attrs.cpu().numpy()
                preds = preds.cpu().numpy()
                # append
                preds_tensor = np.append(preds_tensor, preds, axis=0)
                attrs_tensor = np.append(attrs_tensor, attrs, axis=0)
                # print info
                if (step + 1) % 10 == 0:
                    print('Step: {}/{}'.format(step * labels.size(0), len(self.val_loader)))

        # Evaluation.
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        average_precision = 0.0
        average_recall = 0.0
        average_f1score = 0.0
        valid_count = 0
        for i, name in enumerate(self.attribute_list):
            y_true, y_pred = attrs_tensor[:, i], preds_tensor[:, i]
            accuracy_list.append(accuracy_score(y_true, y_pred))
            if check_metric_valid(y_pred, y_true):  # exclude ill-defined cases
                precision_list.append(precision_score(y_true, y_pred, average='binary'))
                recall_list.append(recall_score(y_true, y_pred, average='binary'))
                f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
                average_precision += precision_list[-1]
                average_recall += recall_list[-1]
                average_f1score += f1_score_list[-1]
                valid_count += 1
            else:
                precision_list.append(-1)
                recall_list.append(-1)
                f1_score_list.append(-1)

        average_acc = np.mean(accuracy_list)
        average_precision = average_precision / valid_count
        average_recall = average_recall / valid_count
        average_f1score = average_f1score / valid_count

        ######################################################################
        # Print
        # ---------
        print("\n"
              "The Precision, Recall and F-score are ignored for some ill-defined cases."
              "\n")

        from prettytable import PrettyTable
        table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
        for i, name in enumerate(self.attribute_list):
            table.add_row([name,
                           '%.3f' % accuracy_list[i],
                           '%.3f' % precision_list[i] if precision_list[i] >= 0.0 else '-',
                           '%.3f' % recall_list[i] if recall_list[i] >= 0.0 else '-',
                           '%.3f' % f1_score_list[i] if f1_score_list[i] >= 0.0 else '-',
                           ])
        print(table)

        print('Average accuracy: {:.4f}'.format(average_acc))
        # print('Average precision: {:.4f}'.format(average_precision))
        # print('Average recall: {:.4f}'.format(average_recall))
        print('Average f1 score: {:.4f}'.format(average_f1score))

        # Save results.
        result = {
            'average_acc': average_acc,
            'average_f1score': average_f1score,
            'accuracy_list': accuracy_list,
            'precision_list': precision_list,
            'recall_list': recall_list,
            'f1_score_list': f1_score_list,
        }
        json.dump(open(os.path.join(self.save_dir, 'acc.mat'), 'w'), result)

