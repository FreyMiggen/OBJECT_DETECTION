# RE-IMPLEMENT FCOS AND FASTER-CNN PAPER


## 1. Backbone and Feature Pyramid Network (FPN)

First, we start building the backbone and FPN of our detector (blue and green parts above). It is the core component that takes in an image and outputs its features of different scales. It can be any type of convolutional network that progressively downsamples the image (e.g. via intermediate max pooling).
Here, we use a small [RegNetX-400MF](https://pytorch.org/vision/stable/models.html#torchvision.models.regnet_x_400mf) as the backbone so we can train in reasonable time on Colab. We initialize this backbone from pre-trained ImageNet weights and extract intermediate features `(c3, c4, c5)` as shown in the figure above.
These features `(c3, c4, c5)` have height and width that is ${1/8}^{th}$, ${1/16}^{th}$, and ${1/32}^{th}$ of the input image respectively.
These values `(8, 16, 32)` are called the "stride" of these features.
In other words, it means that moving one location on the FPN level is equivalent to moving `stride` pixels in the input image.
We add some convolutional block to turn this backbone in to FPN like in this following figure:
<img src="https://github.com/FreyMiggen/OBJECT_DETECTION/tree/d8b7f08061d954166f12a13df9203b13d4794644/resource/feature_pyramid.png" alt="Feature Pyramid Model">

<p align="center" float='left'>
  <img src="resource/feature_pyramid.png" width="400" />
    <img src="resource/feature_pyramid.png" width="400" />
</p>

For more details, see Figure 3 in [FPN paper](https://arxiv.org/abs/1612.03144).
FPN will convert these `(c3, c4, c5)` multi-scale features to `(p3, p4, p5)`. These notations "p3", "p4", "p5" are called feature maps at different level.

## 2. Fully-Convolutional One-Stage Object Detection

FCOS is a fully-convolutional one-stage object detection model — unlike two-stage detectors like Faster R-CNN, it does not comprise any custom modules like anchor boxes, RoI pooling/align, and RPN proposals (for second stage). Due to its simplicity, you will implement core components of FCOS in this first half of the assignment, and then re-use many of them to implement Faster R-CNN in the second half.

An overview of the model in shown below. In case it does not load, see [Figure 2 in FCOS paper](https://arxiv.org/abs/1904.01355).
It details three modeling components: backbone, feature pyramid network (FPN), and head (prediction layers).
First, we will implement FCOS as shown in this figure, and then implement components to train it with the PASCAL VOC 2007 dataset we loaded above.

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-23_at_3.34.09_PM_SAg1OBo.png" alt="FCOS Model Figure" width="80%">

> **CAUTION:** The original FCOS model (as per figure above, and lecture slides) places the centerness predictor in parallel with classification predictor. However, we will follow the widely prevalent implementation practice to place the centerness predictor in parallel with box regression predictor.
The main intuition is that centerness and box regression are localization-related quantities and hence would benefit to have shared features.
