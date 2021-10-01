# Fully Convolutional Networks (FCN) for Apple Segmentation
This project performs Apple Segmentation on MinnieApple dataset using Fully Convolutional Networks (FCN).

## Evaluation
Here are the results on Apple Segmentation task.

### Apple Segmentation
| Method | Backbone | IoU | Class IoU (Apple) |  Pixel Accuracy | Class Accuracy (Apple) |
|---|---|---|---|---|---|
| FCN-ResNet50 (COCO pretraining)  |  ResNet50  |  0.635 | 0.341 | 0.968  | 0.455  | 
| FCN-ResNet101 (COCO pretraining) |  ResNet101  |  0.649 |  0.455 | 0.959  | 0.634  | 
| FCN-ResNet50 (no pretraining)  | ResNet50  | 0.678  |  0.397 | 0.960  | 0.818  | 
| FCN-ResNet101 (no pretraining)  | ResNet101  | 0.685  |  0.410 | 0.962  | 0.848  | 

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
To train a FCN model on the MinneApple dataset, download the dataset first from [here](https://conservancy.umn.edu/handle/11299/206575).
* Now, you can train a FCN-ResNet50 or FCN-ResNet101 network with the following commands.
```
# Train FCN-ResNet50
python train_fcn.py --data_path /path/to/MinneApple/dataset --model fcn_resnet50 --epochs 64 --output-dir /path/to/checkpoint/directory
# Train FCN-ResNet101
python train_fcn.py --data_path /path/to/MinneApple/dataset --model fcn_resnet101 --epochs 64 --output-dir /path/to/checkpoint/directory

```

## Prediction 
To use a model for prediction run the following command:
```
# Predict for FCN-ResNet50
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --fcn_resnet50
# Predict for FCN-ResNet101
python predict_fcn.py --data_path /path/to/MinneApple/dataset --output_path /path/to/which/to/write/the/predictions --weight_file /path/to/a/weight/file --device [one out of: cpu/gpu] --fcn_resnet101
```

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
