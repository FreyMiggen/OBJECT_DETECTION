# RE-IMPLEMENT FCOS AND FASTER-CNN PAPER


## 1. Backbone and Feature Pyramid Network (FPN)

First, we start building the backbone and FPN of our detector (blue and green parts above). It is the core component that takes in an image and outputs its features of different scales. It can be any type of convolutional network that progressively downsamples the image (e.g. via intermediate max pooling).
Here, we use a small [RegNetX-400MF](https://pytorch.org/vision/stable/models.html#torchvision.models.regnet_x_400mf) as the backbone so we can train in reasonable time on Colab. We initialize this backbone from pre-trained ImageNet weights and extract intermediate features `(c3, c4, c5)` as shown in the figure above.
These features `(c3, c4, c5)` have height and width that is ${1/8}^{th}$, ${1/16}^{th}$, and ${1/32}^{th}$ of the input image respectively.
These values `(8, 16, 32)` are called the "stride" of these features.
In other words, it means that moving one location on the FPN level is equivalent to moving `stride` pixels in the input image.
We add some convolutional block to turn this backbone in to FPN like in this following figure:

<p>
  <img src="resource/feature_pyramid.png" width="80%" />

</p>

For more details, see Figure 3 in [FPN paper](https://arxiv.org/abs/1612.03144).
FPN will convert these `(c3, c4, c5)` multi-scale features to `(p3, p4, p5)`. These notations "p3", "p4", "p5" will, from now on, called feature maps at different levels.
Some more detail on how to implement FPN from backbone network:

```
# Create three separate 1x1 convolutional layer to perform 'lateral connection' (lateral arrow in figure 3)
        la3=nn.Conv2d(out_channels_c3,out_channels,(1,1))
        la4=nn.Conv2d(out_channels_c4,out_channels,(1,1))
        la5=nn.Conv2d(out_channels_c5,out_channels,(1,1))

# After extract multi-scale features c3,c4,c5 from backbone network (look at figure above to understand logic behind this implementation)
        p5=la5(c5)

        p4=la4(c4)
        p5_up=F.interpolate(p5,size=c4.shape[2:])  
        p4=p4+p5_up

        p3=la3(c3)
        p4_up=F.interpolate(p4,c3.shape[2:])
        p3=p3+p4_up

```
In reality, after obtaining p3,p4,p5; we will pass these through additional conv layers for better performance.
## 2. Fully-Convolutional One-Stage Object Detection

FCOS is a fully-convolutional one-stage object detection model — unlike two-stage detectors like Faster R-CNN, it does not comprise any custom modules like anchor boxes, RoI pooling/align, and RPN proposals (for second stage).

An overview of the model in shown below. In case it does not load, see [Figure 2 in FCOS paper](https://arxiv.org/abs/1904.01355).
It details three modeling components: backbone, feature pyramid network (FPN), and head (prediction layers). Two of those are explained in Section 1.
First, we will implement FCOS as shown in this figure, and then implement components to train it with the PASCAL VOC 2007 dataset we loaded above.

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-23_at_3.34.09_PM_SAg1OBo.png" alt="FCOS Model Figure" width="80%">

> **CAUTION:** The original FCOS model (as per figure above, and lecture slides) places the centerness predictor in parallel with classification predictor. However, we will follow the widely prevalent implementation practice to place the centerness predictor in parallel with box regression predictor.
The main intuition is that centerness and box regression are localization-related quantities and hence would benefit to have shared features.
