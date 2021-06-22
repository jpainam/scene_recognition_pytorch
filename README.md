#### [Paper] Scene recognition with objectness, attribute and category learning
![The Overall framework](./imgs/framework.png)
### Datasets
The annotated datasets can be found in `annotations` folder. 
We mined 134 attributes for 4 datasets :
* <a href="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip">ADE20K</a>
* <a href="http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar">Places365</a>
* <a href="http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/MITIndoor67.zip">MITIndoor67</a> 
* <a href="http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/SUN397.zip">SUN397</a> 

The attributes are annotated at the image level, 
thus each `annotation` file contains 

```
dict() --> dict_keys([images, categories, attributes, labels])
    -- images: list() size = num_images
        list of image names (eg. airport_inside_0344.jpg)
    -- categories: list() size = num_images
        list of categories (eg. airport_inside)
    -- attributes: list() size = 134
        the list of 134 object attributes (eg. person, wall, cup, bottle etc...)
    -- labels: matrix size = num_images x 134
         Row index of this matrix -> images[row_index] and categories[row_index]
```

The annotations for training are contained in the file named `train_annotation.pkl`
 and testing/validation annotations are contained in `val_annotations.pkl` 


Object attribute  | Representation | label
------------- | ------------- | -----------
person | confidence score | 0.78897
bicycle | confidence score | 0.81421 
car | confidence score | 0   
wall |  presence score | no(0), yes(1) 
sky | presence score | no(0), yes(1)
etc. | ... | ...
### Configuration

Define the params in the configuration file located in the `config` folder
```
DATASET:
  NAME: SUN397
  ROOT: /home/fstd/datasets/SUN397
  TRAIN: train
  VAL: val
  NUM_CATEGORY: 397

MODEL:
  ARCH: 50 
  NUM_FEATURES: 2048 
  BACKBONE: resnext
  WITH_ATTRIBUTE: True
  ARM: True

TRAINING:
  EPOCH: 100
  BATCH_SIZE: 16

TESTING:
  BATCH_SIZE: 8
  CHECKPOINT: best # best | 100 | 90 etc...
  TEN_CROPS: True
```
```
ARCH =  50, 101 for resnet
        19 for vgg
        161 for densenet
NUM_FEATURES =  2208 for densenet
                512 for vgg
                2048 for resnet | resnext
BACKBONE = vgg | resnet | resnext | densenet
```
### Training
```
python main.py --config config/MITIndoor67.yaml
```

### Testing
```
python test.py --config config/MITIndoor67.yaml
```

#### Results for scene recognition
* MIT67 dataset

Model | Top@1 | Top@5 | Top@10 
------| ----- | ----- | ----- 
MASR-VGG | 76.9  |  88.0 | 94.6 
MASR-ResNet50  |  86.2  |  94.8 |  98.9 
MASR-ResNext101  |  88.5  | 95.3  | 98.7 

* SUN937 dataset

Model | Top@1 | Top@5 | Top@10 
------| ----- | ----- | ----- 
MASR-VGG | 60.1  | 75.2  | 86.7
MASR-ResNext101  | 75.01  | 86.8 | 94.6  

* ADE20K dataset

Model | Top@1 | Top@5 | Top@10 
------| ----- | ----- | -----     
MASR-VGG | 61.3 | 72.5 | 80.4
MASR-ResNext101  | 64.4  | 75.2 | 85.1

#### Attribute accuracy for SUN397
```
person              77.4
floor-wood          100.0
house               82.8
playingfield        100.0
river               93.9
road                89.3
sea                 93.0
shelf               82.6
snow                100.0
wall-wood           100.0
window              79.7
tree                92.5
fence               84.4
ceiling             87.1
sky                 95.7
cabinet             92.5
floor               73.1
pavement            76.2
mountain            88.0
grass               89.3
dirt                88.3
building            83.0
rock                93.8
wall                87.1
AP: 0.8873361234281417
```