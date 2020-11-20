import torch
from models.resnet import *
from evaluation.evaluation import Evaluation
from data.data_manager import get_data
import argparse
import yaml
import os.path as osp

parser = argparse.ArgumentParser(description='Scene Recognition Training Procedure')
parser.add_argument('--config', metavar='DIR', help='Configuration file path')
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.config, 'r'))

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_data(root=CONFIG['DATASET']['ROOT'])
    model = resnet50(pretrained=True, num_classes=len(class_names), num_features=512)
    # Load the last epoch checkpoint
    checkpoint = osp.join(CONFIG['MODEL']['OUTPUT'], 'model_{}.pth.tar'.format(CONFIG['TESTING']['CHECKPOINT']))
    print()
    print("Checkpoint loaded!")
    model.load_state_dict(torch.load(checkpoint))
    if torch.cuda.is_available():
        model = model.cuda()
    evaluate = Evaluation(model=model,  dataloader=val_loader, classes=class_names)
    evaluate.test(topk=(1, 2, 5))