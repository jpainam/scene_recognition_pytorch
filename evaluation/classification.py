from __future__ import absolute_import

import torch
from easydict import EasyDict
import numpy as np
from sklearn.metrics import precision_score


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def precision(outputs, labels):
    op = outputs.cpu()
    la = labels.cpu()
    _, preds = torch.max(op, dim=1)
    return torch.tensor(precision_score(la, preds, average='weighted'))


def accuracy(output, target, topk=(1,), is_multilabel=False):
    if not is_multilabel:
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].contiguous().view(-1).float().sum(0) * 100.0 / batch_size for k in topk]
    else:
        raise Exception('set is_multilabel=False, unimplemented multilabel')


def getclassAccuracy(output, target, nclasses, topk=(1,)):
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
        ClassAccuracy = torch.zeros([1, nclasses], dtype=torch.uint8).cuda()
        correct_k = correct[:, :k].sum(1)
        for n in range(target.shape[0]):
            ClassAccuracy[0, target[n]] += correct_k[n].byte()
        ClassAccuracyRes.append(ClassAccuracy)

    return ClassAccuracyRes



def accuracy2(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    print(pred.shape)
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / target.sum())
    return ret

'''

def accuracy(output, target, topk=(1,)):
    from ignite.metrics import Accuracy, Precision, Recall
    acc = Accuracy(is_multilabel=True)
    acc.reset()
    for y_pred, y in zip(output, target):
        acc.update((y_pred, y))

    precision = Precision(average=False, is_multilabel=True)
    recall = Recall(average=False, is_multilabel=True)
    return acc.compute()

'''
def accuracy_multilabel(output, attrs):
    # computes the total accuracy for nn.nn.MultiLabelSoftMarginLoss

    vqa_score = 0
    _, oix = output.data.max(1)
    for i, pred in enumerate(oix):
        count = attrs[i,pred]

        vqa_score += min(count /3, 1)
    return vqa_score


# computes the total accuracy for nn.nn.MultiLabelSoftMarginLoss
def accuracy_multilabel2(outputs, labels):
    N, C = outputs.size()
    #outputs = torch.sigmoid(outputs)  # torch.Size([N, C]) e.g. tensor([[0., 0.5, 0.]])
    #outputs[outputs >= 0.5] = 1.
    #outputs[outputs < 0.5] = 0.
    preds = torch.gt(outputs, torch.ones_like(outputs) / 2.)
    acc = (preds.bool() == labels.bool()).sum() / (N * C) * 100
    return acc
'''
class LabelwiseAccuracy(Accuracy):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_examples = None
        super(LabelwiseAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = None
        self._num_examples = 0
        super(LabelwiseAccuracy, self).reset()

    def update(self, output):

        #y_pred, y = self._check_shape(output)
        self._check_shape(output)
        y_pred, y = output
        self._check_type((y_pred, y))

        num_classes = y_pred.size(1)
        last_dim = y_pred.ndimension()
        y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
        y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
        correct_exact = torch.all(y == y_pred.type_as(y), dim=-1)  # Sample-wise
        correct_elementwise = torch.sum(y == y_pred.type_as(y), dim=0)

        if self._num_correct is not None:
            self._num_correct = torch.add(self._num_correct,
                                                    correct_elementwise)
        else:
            self._num_correct = correct_elementwise
        self._num_examples += correct_exact.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._num_correct.type(torch.float) / self._num_examples
'''


def get_attribute_results(gt_label, preds_probs, thresold=0.5):
    pred_label = preds_probs > thresold
    eps = 1e-20
    result = EasyDict()

    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

