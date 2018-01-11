---
title: Segmenting complex cells with CNNs
permalink: /segmentation-fully-connected
status: published
---


This post is a case study of how modern “Deep Learning” techniques can help biologists study living cells. It describes how convolutional neural networks can automate data analysis, by segmenting images of cells taken from time-lapse live-cell microscopy images. Deep learning makes it possible to quantitatively analyze complex cell images that confound traditional computer vision techniques.

<br>
<br>
*Background*

Biologists study living cells to understand how they behave, both in natural growth and development and during disease. Individual cells often exhibit diverse behaviors - think about how a single fertilized egg cell differentiates into the wide variety of cells in the human body. Light microscopy provides an especially powerful tool to study individual living cells over time, letting us watch these cells directly over time.

Computational image segmentation helps us automatically extract cell information from microscopy images. This information can be as simple as cell size, whose change over time tells us about growth. Or it could be more complex, like signals from genetically encoded fluorescent reporters, which tell us how strongly various genes are being expressed. Regardless of the information, accurate and high-speed image segmentation is crucial to creating high throughput image analysis pipelines, making the science much more productive and much more fun.

Is it easy to write a computer program to find cells in microscopy images? It depends. Some images, such as phase-contrast images of rod-shaped bacteria, are easy to segment. These images contain cells clearly separated from background and from each other, with regular shapes and intensities. Other images, like those of embryonic stem cells in certain conditions, are really tough. These images have cells with widely varying, irregular morphologies and intensities. The images are so irregular that traditional segmentation methods fail completely on them.

So how can we segment these difficult cells?
<div style="text-align:center">
<img src="assets/2018-01-03-im01.png" />
<figcaption><b>Our (difficult) task.</b> Segment individual mouse embryonic stem (mES) cells in this image.</figcaption>
</div>
<br>

<br>
<br>

*Method*

The bet is that modern, multi-layer (deep) neural network architectures (DNNs) can help solve this segmentation problem.

Using convolutional DNNs (or CNNs) to segment images is well established. At a high level, the problem is similar to image classification, except that one is often classifying individual pixels instead of entire images. Methods range from semantic segmentation to instance segmentation. Initial work in semantic segmentation directly repurposed classification techniques to segment individual pixels by classifying their surrounding image patches. Unfortunately this method was extremely slow. Recent techniques use a “fully convolutional” approach for both training and inference. This avoids the redundant computation of the patch-wise approach and harvests significant speed gains.

To create our training set, we first hand segmented seven (yes, only seven!) ES-cell images.
<br>
<div style="text-align:center">
<img src="assets/2018-01-03-im02.png">
<figcaption><b>Training mES images.</b> (Left) Original image. (Center) Cell masks. (Right) cell edges.</figcaption>
</div>
<br>


We then augmented this data set by selecting sub-images from our original 1024x1024 images and applying random x and y circular translations (‘rolls’), flips, and rotations. Each augmented image serves as a new piece of training data, so this process creates thousands of small, randomly augmented training images from our original set of seven large images.
<br>
<div style="text-align:center">
<img src="assets/2018-01-03-im03.png">
<figcaption><b>Augmenting training images.</b> (Left) Original image. (Right) After augmentation.</figcaption>
</div>
<br>


While initial experiments with relatively small networks readily segmented ‘easy’ bacterial images, they had difficulty segmenting more irregular ES images. We hypothesized the problem was that the networks were too simple, and that 'higher capacity' networks might be able to capture all the features necessary for this more complex segmentation task. We chose GoogLeNet for its higher capacity and reasonable resource requirements.

To adapt GoogLeNet for semantic segmentation, we replaced the final fully connected output described in the original GoogLeNet paper with a single convolution-transpose layer. This layer restores the reduced representation created by GoogLeNet’s multi-stride convolutional and max-pool layers to the original size. We trained on the augmented data set using RMSprop with an annealed learning rate, using an Amazon p2.8xlarge GPU instance.

The network performs extremely well on this task, with individual cells clearly defined and separated from one another.
<br>
<div style="text-align:center">
<img src="assets/2018-01-03-im04.png" />
<figcaption><b>Segmentation results.</b> mES cells segmented by fully convolutional GoogLeNet architecture. (Left) Original image. (Center) Segmented mask. (Right) Composite of original and mask. Composite pixel colors range continuously from blue (non-cell) to green (undecided) to red (cell).</figcaption>
</div>
<br>
<br>
<div style="text-align:center">
<img src="assets/2018-01-03-im05.png" />
<figcaption><b>Segmentation results (close-up).</b> mES cells segmented by fully convolutional GoogLeNet architecture. (Left) Original image. (Center) Segmented mask. (Right) Composite of original and mask. Composite pixel colors range continuously from blue (non-cell) to green (undecided) to red (cell).</figcaption>
</div>
<br>



One challenge with ES cell images is poorly defined separation between cells. To address, we overweight errors made on edge pixels during training. This encourages the network to be careful about how it treats edges, resulting in good separations between cells.

Converting the network's output into a segmentation mask is straightforward, using simple thresholding and connected component operations. A pixel-by-pixel ROC analysis of out-of-sample images demonstrates that simple thresholding creates a good classifier.

<div style="text-align:center">
<img src="assets/2018-01-03-im06.png" />
<figcaption><b>From image to mask via segmentation and thresholding.</b> (Upper left) Original fluorescence image. (Upper right) Class probabilities produced by GoogLeNet. (Lower left) Cell mask after thresholding and region labeling. (Lower right) Pixelwise ROC curves demonstrate that good thresholds exist. Blue curve  calculated from a training image; orange curve from a test image. </figcaption>
</div>
<br>

<br>
<br>

*Conclusion*

High capacity convolutional neural networks like GoogLeNet, incorporated into a simple fully convolutional architecture, effectively segment light microscopy cell images. This method works well even with highly irregular cell images. Just as importantly, it's practical. Hand segmentation of the seven training images took 1-2 days. With our standard hardware set up, training takes approximately 1 hour and yields consistently reproducible results, while inference (segmentation) takes less than 1 second per frame. Whereas before, figuring out how to deal with difficult new experimental conditions (new cell types, and/or new environmental set up) could be a show stopper, now a new pipeline can be trained on these conditions with less than a week's work.

A few natural next steps suggest themselves. First, a small fraction of the segmented cells sometimes need to be edited. Multiple cells are sometimes connected, and large cells are sometimes split in two. More sophisticated instance segmentation methods could potentially solve these problems. Also, it would be interesting to understand *how* these larger networks segment, by understanding their internal representations, and how these representations change from architecture to architecture. These challenges will hopefully be taken up by those in the field.
