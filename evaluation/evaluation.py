from utils.meters import AverageMeter
import numpy as np
import torch
import time
from .classification import accuracy
from torch import nn
from torch.backends import cudnn

class Evaluation(object):
    def __init__(self, model, dataloader, classes, ten_crops, with_attribute=False):
        self.model = model
        self.model.eval()
        self.dataloader = dataloader
        self.classes = classes
        self.use_cuda = torch.cuda.is_available()
        cudnn.benchmark = self.use_cuda
        self.criterion = nn.CrossEntropyLoss()
        self.ten_crops = ten_crops
        self.with_attribute = with_attribute

    def test(self, topk=(1,)):
        # Evaluate model on validation set

        val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = self.__eval(topk)

        # Save Validation Class Accuracy
        val_ClassAcc_top1 = (val_ClassTPDic['Top1'] / len(self.classes)) * 100
        np.savetxt('ValidationTop1ClassAccuracy.txt', np.transpose(val_ClassAcc_top1),'%f')

        val_ClassAcc_top2 = (val_ClassTPDic['Top2'] /len(self.classes)) * 100
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
        Predictions = np.zeros(len(self.dataloader))
        SceneGTLabels = np.zeros(len(self.dataloader))

        # Start data time
        data_time_start = time.time()

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(self.dataloader):
                start_time = time.time()
                if self.use_cuda:
                    images, labels = images.cuda(), labels.cuda()
                    # attributes = attributes.cuda() if self.with_attribute else None

                if self.ten_crops:
                    bs, ncrops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                outputs, attrs, features = self.model(images)
                if self.ten_crops:
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                loss = self.criterion(outputs, labels)
                prec1, prec2, prec5 = accuracy(outputs.data, labels.data, topk)

                # ten_predictions = self.__kTopPredictedClasses(10, outputs)
                classTps = self._getClassAccuracy(outputs, labels, topk)

                ClassTPs_Top1 += classTps[0]
                ClassTPs_Top2 += classTps[1]
                ClassTPs_Top5 += classTps[2]

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
            return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic

    def _kTopPredictedClasses(self, kth, predictions):
        _, preds = predictions.topk(kth, largest=True, sorted=True)
        idx = preds.cpu().numpy()[0]
        return idx

    def _getClassAccuracy(self,output, target, topk=(1,)):

        """
        Computes the top-k accuracy between output and target and aggregates it by class
        :param output: output vector from the network
        :param target: ground-truth
        :param nclasses: nclasses in the problem
        :param topk: Top-k results desired, i.e. top1, top2, top5
        :return: topk vectors aggregated by class
        """
        maxk = max(topk)

        score, label_index = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        correct = label_index.eq(torch.unsqueeze(target, 1))

        ClassAccuracyRes = []
        for k in topk:
            ClassAccuracy = torch.zeros([1, len(self.classes)], dtype=torch.uint8).cuda()
            correct_k = correct[:, :k].sum(1)
            for n in range(target.shape[0]):
                ClassAccuracy[0, target[n]] += correct_k[n].byte()
            ClassAccuracyRes.append(ClassAccuracy)

        return ClassAccuracyRes