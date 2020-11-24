from .resnet import *


def create(num_features, num_classes):
    model = resnet50(pretrained=True, num_classes=num_classes, num_features=int(num_features))
    print()
    print("Model Loaded:")
    print(f"num_features : {num_features}")
    return model