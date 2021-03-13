import torch
from models.resnet import *
from evaluation.evaluation import Evaluation
from data.data_manager import get_data
import argparse
import yaml
import os.path as osp
from torch import nn
import models

parser = argparse.ArgumentParser(description='Scene Recognition Training Procedure')
parser.add_argument('--config', metavar='DIR', help='Configuration file path')
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.config, 'r'))
with_attribute = CONFIG['TRAINING']['WITH_ATTRIBUTE']

if __name__ == "__main__":
    train_loader, val_loader, class_names, attrs = get_data(dataset=CONFIG['DATASET']['NAME'],
                                                            root=CONFIG['DATASET']['ROOT'],
                                                            val_folder='val',
                                                            train_folder='train',
                                                            batch_size=CONFIG['TESTING']['BATCH_SIZE'],
                                                            ten_crops=CONFIG['TESTING']['TEN_CROPS'],
                                                            with_attribute=with_attribute)
    model = models.get_model(with_attribute=with_attribute,
                             num_features=CONFIG['MODEL']['NUM_FEATURES'],
                             num_classes=len(class_names),
                             num_attrs=len(attrs),
                             dropout=0.5)
    # Load the last epoch checkpoint
    checkpoint = osp.join(CONFIG['MODEL']['CHECKPOINTS'], CONFIG['DATASET']['NAME'],
                          'model_{}.pth.tar'.format(CONFIG['TESTING']['CHECKPOINT']))
    print()
    print(f"Checkpoint:  {checkpoint}!")
    print("Checkpoint loaded!! ")
    pretrained_dict = torch.load(checkpoint)
    #pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    evaluate = Evaluation(model=model, dataloader=val_loader, classes=class_names,
                          ten_crops=CONFIG['TESTING']['TEN_CROPS'],
                          with_attribute=CONFIG['TRAINING']['WITH_ATTRIBUTE'])
    evaluate.test(topk=(1, 2, 5))
