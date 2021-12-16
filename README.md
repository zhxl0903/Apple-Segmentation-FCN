# Fully Convolutional Networks for Apple Segmentation
This project performs Apple Segmentation task on MinneApple dataset using some fully convolutional models including Fully Convolutional Networks (FCN) and DeeplabV3.

## Evaluation
Here are the results on Apple Segmentation task.

### Apple Segmentation
| Method                                                         | Backbone | IoU   | Class IoU (Apple) | Pixel Accuracy | Class Accuracy (Apple) |
|----------------------------------------------------------------|---|-------|-------------------|----------------|------------------------|
| FCN-ResNet50 (no pretraining)                                  | ResNet50  | 0.788 | 0.592             | 0.984          | 0.808                  | 
| FCN-ResNet50 (ImageNet pretraining for ResNet backbone)        | ResNet50  | 0.815 | 0.644             | 0.987          | 0.844                  | 
| FCN-ResNet50 (COCO pretraining)                                |  ResNet50  | 0.811 | 0.636             | 0.986          | 0.839                  | 
| FCN-ResNet101 (no pretraining)                                 | ResNet101  | 0.792 | 0.600             | 0.984          | 0.823                  | 
| FCN-ResNet101 (ImageNet pretraining for ResNet backbone)       | ResNet101  | 0.806 | 0.628             | 0.985          | 0.842                  | 
| FCN-ResNet101 (COCO pretraining)                               |  ResNet101  | 0.808 | 0.630             | 0.986          | 0.835                  |
| DeeplabV3-ResNet50 (no pretraining)                            | ResNet50  | 0.784 | 0.584             | 0.983          | 0.819                  | 
| DeeplabV3-ResNet50 (ImageNet pretraining for ResNet backbone)  | ResNet50  | 0.806 | 0.627             | 0.986          | 0.821                  | 
| DeeplabV3-ResNet50 (COCO pretraining)                          |  ResNet50  | 0.821 | 0.654             | 0.987          | 0.833                  | 
| DeeplabV3-ResNet101 (no pretraining)                           | ResNet101  | 0.799 | 0.613             | 0.986          | 0.786                  | 
| DeeplabV3-ResNet101 (ImageNet pretraining for ResNet backbone) | ResNet101  | 0.804 | 0.654             | 0.987          | 0.833                  | 
| DeeplabV3-ResNet101 (COCO pretraining)                         |  ResNet101  | 0.821 | 0.621             | 0.987          | 0.786                  |
| Semi-supervised GMM                                            |  -  | 0.635 | 0.341             | 0.968          | 0.455                  | 
| User-supervised GMM                                            |  -  | 0.649 | 0.455             | 0.959          | 0.634                  | 
| UNet (no pretraining)                                          | ResNet50  | 0.678 | 0.397             | 0.960          | 0.818                  | 
| UNet (ImageNet pretraining)                                    | ResNet50  | 0.685 | 0.410             | 0.962          | 0.848                  | 

## Installation
* Setup a Python 3 environment
* Install Pytorch (1.0.1 or higher) and TorchVision
* Install some other packages:
```
# Install necessary packages
pip install Pillow opencv-python sklearn numpy
```
* Clone this repository and enter it
```
git clone https://github.com/zhxl0903/Apple-Segmentation-FCN.git
cd Apple-Segmentation-FCN
```

## Training
To train a FCN or DeeplabV3 model on the MinneApple dataset, download the dataset first from [here](https://conservancy.umn.edu/handle/11299/206575).
* Now, you can train a FCN-ResNet50, FCN-ResNet101, DeeplabV3-ResNet50, or DeeplabV3-ResNet101 network with the following commands.
```
# Train FCN-ResNet50
python train_fcn.py --data_path /path/to/MinneApple/dataset --model fcn_resnet50 --epochs 64 --output-dir /path/to/checkpoint/directory
# Train FCN-ResNet101
python train_fcn.py --data_path /path/to/MinneApple/dataset --model fcn_resnet101 --epochs 64 --output-dir /path/to/checkpoint/directory
# Train FCN-ResNet50
python train_fcn.py --data_path /path/to/MinneApple/dataset --model deeplabv3_resnet50 --epochs 64 --output-dir /path/to/checkpoint/directory
# Train FCN-ResNet101
python train_fcn.py --data_path /path/to/MinneApple/dataset --model deeplabv3_resnet101 --epochs 64 --output-dir /path/to/checkpoint/directory
```
* DeeplabV3 models require a batch size greater than 1 during training. You can set the batch size using argument: --batch-size n (Here, n is the batch size)
* DeeplabV3 models require the size of the last minibatch to be greater than 1. You can drop the last minibatch using argument: --drop_last_batch
* To train the model with Imagenet pretrained backbone, use the argument: --imagenet_pretrained_backbone
* To train the model with COCO pretraining, use the argument: --pretrained

## Prediction 
To use a model for prediction run the following command:
```
# Predict for FCN-ResNet50
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --fcn_resnet50
# Predict for FCN-ResNet101
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --fcn_resnet101
# Predict for DeeplabV3-ResNet50
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --deeplabv3_resnet50
# Predict for DeeplabV3-ResNet101
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --deeplabv3_resnet101
```

## Evaluation on MinneApple Test Set
Evaluation of network on test set of MinneApple dataset can be performed on the CodaLab competition: [fruit segmentation](https://competitions.codalab.org/competitions/21694).

Submit outputs from predict_fcn.py on test set in a zipfile, results.zip, to the CodeLab competition.

## References

```
@misc{hani2019minneapple,
    title={MinneApple: A Benchmark Dataset for Apple Detection and Segmentation},
    author={Nicolai Häni and Pravakar Roy and Volkan Isler}
    year={2019},
    eprint={1909.06441},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
