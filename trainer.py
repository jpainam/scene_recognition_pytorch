import time
from utils.meters import AverageMeter
import torch
import numpy as np
from tqdm import tqdm
from utils import time_str


class BaseTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, config, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

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
        epoch_time = time.time()
        batch_num = len(self.train_loader)
        gt_list = []
        preds_probs = []
        loss_meter = AverageMeter()
        lr = self.optimizer.param_groups[1]['lr']

        for step, (imgs, gt_label) in enumerate(self.train_loader):
            batch_time = time.time()
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            #gt_label[gt_label != 0] = 1.
            #gt_label = gt_label.type(torch.cuda.LongTensor)

            logits_attrs, _ = self.model(imgs)
            loss = self.criterion(logits_attrs, gt_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())
            gt_list.append(gt_label.detach().cpu().numpy())
            train_probs = torch.sigmoid(logits_attrs)
            preds_probs.append(train_probs.detach().cpu().numpy())

            log_interval = 20
            if (step + 1) % log_interval == 0 or (step + 1) % len(self.train_loader) == 0:
                print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                      f'train_loss:{loss_meter.val:.4f}')

        train_loss = loss_meter.avg
        gt_label = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)

        print(f'Epoch {epoch} LR {lr}, Time train {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

        return train_loss, gt_label, preds_probs


    def eval(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()

        preds_probs = []
        gt_list = []
        with torch.no_grad():
            for step, (imgs, gt_label) in enumerate(tqdm(self.val_loader)):
                imgs = imgs.cuda()
                gt_label = gt_label.cuda()
                gt_list.append(gt_label.cpu().numpy())
                gt_label[gt_label == -1] = 0
                valid_logits, _ = self.model(imgs)
                valid_loss = self.criterion(valid_logits, gt_label)
                valid_probs = torch.sigmoid(valid_logits)
                preds_probs.append(valid_probs.cpu().numpy())
                loss_meter.update(valid_loss.item())

        valid_loss = loss_meter.avg

        gt_label = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)
        return valid_loss, gt_label, preds_probs
