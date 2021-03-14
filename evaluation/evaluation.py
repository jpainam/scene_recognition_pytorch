from utils.meters import AverageMeter
import numpy as np
import torch
import time
from .classification import accuracy, getclassAccuracy
from torch import nn
from torch.backends import cudnn
from sklearn import metrics
from sklearn.metrics import top_k_accuracy_score


class Evaluation(object):
    def __init__(self, model, dataloader, classes, ten_crops,
                 with_attribute=False, xi=.8):
        self.model = model
        self.model.eval()
        self.dataloader = dataloader
        self.classes = classes
        self.use_cuda = torch.cuda.is_available()
        cudnn.benchmark = self.use_cuda
        self.criterion = nn.CrossEntropyLoss()
        self.ten_crops = ten_crops
        self.with_attribute = with_attribute
        self.xi = xi

    def test(self, topk=(1,)):
        self.model.eval()
        # Evaluate model on validation set

        val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = self.__eval(topk)

        # Save Validation Class Accuracy
        val_ClassAcc_top1 = (val_ClassTPDic['Top1'] / len(self.classes)) * 100
        np.savetxt('ValidationTop1ClassAccuracy.txt', np.transpose(val_ClassAcc_top1), '%f')

        val_ClassAcc_top2 = (val_ClassTPDic['Top2'] / len(self.classes)) * 100
        np.savetxt('ValidationTop2ClassAccuracy.txt', np.transpose(val_ClassAcc_top2), '%f')

        val_ClassAcc_top5 = (val_ClassTPDic['Top5'] / len(self.classes)) * 100
        np.savetxt('ValidationTop5ClassAccuracy.txt', np.transpose(val_ClassAcc_top5), '%f')

        # Print complete evaluation information
        print('-' * 65)
        print('Evaluation statistics:')

        # print('Train results     : Loss {train_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
        #       'Mean Class Accuracy {MCA:.3f}'.format(train_loss=train_loss, top1=train_top1, top2=train_top2, top5=train_top5,
        #                                               MCA=np.mean(train_ClassAcc_top1)))

        print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
              'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5,
                                                     MCA=np.mean(val_ClassAcc_top1)))

    def __eval(self, topk):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()
        top5 = AverageMeter()
        ClassTPs_Top1 = torch.zeros(1, len(self.classes), dtype=torch.uint8).cuda()
        ClassTPs_Top2 = torch.zeros(1, len(self.classes), dtype=torch.uint8).cuda()
        ClassTPs_Top5 = torch.zeros(1, len(self.classes), dtype=torch.uint8).cuda()
        y_pred = []
        y_true = []

        # Start data time
        data_time_start = time.time()

        with torch.no_grad():
            for i, (images, labels, orig_attrs) in enumerate(self.dataloader):
                start_time = time.time()
                if self.use_cuda:
                    images, labels, orig_attrs = images.cuda(), labels.cuda(), orig_attrs.cuda()

                attrs = orig_attrs.detach().clone()
                attrs[attrs > self.xi] = 1.
                attrs[attrs <= self.xi] = 0.
                if self.ten_crops:
                    bs, ncrops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                if self.with_attribute:
                    outputs, _ = self.model(images, orig_attrs)
                else:
                    outputs = self.model(images, orig_attrs)

                if self.ten_crops:
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, dim=1)
                y_true = np.append(y_true, labels.cpu().numpy(), axis=0)
                y_pred = np.append(y_pred, predicted.cpu().numpy(), axis=0)

                # Compute class accuracy
                ClassTPs = getclassAccuracy(outputs, labels, len(self.classes), topk)
                ClassTPs_Top1 += ClassTPs[0]
                ClassTPs_Top2 += ClassTPs[1]
                ClassTPs_Top5 += ClassTPs[2]

                # Measure Top1, Top2 and Top5 accuracy
                prec1, prec2, prec5 = accuracy(outputs.data, labels.data, topk)

                losses.update(loss.item(), labels.size(0))
                top1.update(prec1.item(), labels.size(0))
                top2.update(prec2.item(), labels.size(0))
                top5.update(prec5.item(), labels.size(0))

                batch_time.update(time.time() - start_time)
                if (i + 1) % 10 == 0:
                    print('Testing batch: [{}/{}]\t'
                          'Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                          'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                          format(i, len(self.dataloader), batch_time=batch_time, loss=losses,
                                 top1=top1, top2=top2, top5=top5))

            ClassTPDic = {'Top1': ClassTPs_Top1.cpu().numpy(),
                          'Top2': ClassTPs_Top2.cpu().numpy(), 'Top5': ClassTPs_Top5.cpu().numpy()}

            print(
                'Elapsed time for {} set evaluation {time:.3f} seconds'.format(set, time=time.time() - data_time_start))
            print("")
            print(metrics.precision_score(y_true=y_true,
                                          y_pred=y_pred,
                                          average='micro'))
            return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic

    def _kTopPredictedClasses(self, kth, predictions):
        _, preds = predictions.topk(kth, largest=True, sorted=True)
        idx = preds.cpu().numpy()[0]
        return idx
