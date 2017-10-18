---
title: Segmentation 3 - Architectures
permalink: /segmentation-architectures
status: published
---

In the previous segmentation post I demonstrated how to segment *Bacillus subtilis* cells in a microscopy image using deep convolutional neural networks. The careful reader will note several details remained unexplained. :) One key detail - how did I decide to use that particular network architecture?

Simplicity played a key role. This network is one of the simplest possible - it consists only of sequentially connected convolution, RelU, and MaxPool layers feeding into a fully connected layer. While a wide range of network architecture innovations exist, and improvements even on this simple framework are undoubtedly possible (afficionados will note various bells and whistles could be added to prevent overfitting, for example), it's nice to know this bare-bones CNN works well.

Yet even this simple architecture confronts us with some choices. How many convolutional layers do we need to use? How often do we intersperse MaxPool and RelU layers? Do different designs perform differently? This post will sketch some brief thoughts behind these choices.

The key takeaway from this - *architecture affects performance.*
<br>
<br>
<div style="text-align:center">
<img src="assets/2017-09-25-the-architect.jpg" />
<figcaption>Architecture discussions may seem a little dry.</figcaption>
</div>
<br>
<br>

Design constraints dictate design decisions. In order for us to understand the decisions behind our architecture choice, let's first understand the constraints of this problem.

Our goal is to classify each pixel in the input image. We do this by using each pixel's local context, specifically a patch of pixels around it. We run this patch through our CNN, and the CNN spits out a single 3-element probability distribution. This distribution encodes the probabilities for each its output is a single 3-element probability distribution. This means that for each pixel, our CNN needs to map its surrounding pixel-patch to a single 3-element probability distribution. More specifically, the spatial dimension of the input patch (in our case, 31x31 pixels) must get reduced down to a single pixel. This places a constraint on our network - the layers, acting together, must reduce the input patch down to a single pixel.

Recalling the arithmetic by which convolutional and Max-Pool layers reduce image size, we can track how the initial pixel-patch size changes through the layers of our network:
<center>
<table style  = "width:80%">
  <tr>
    <td><b>Layer</b></td>
    <td><b>Description</b></td>
    <td><b>Output Size</b></td>
  </tr>
  <tr>
    <td>Input</td>
    <td>Pixel-patch</td>
    <td>31x31x1 pixels</td>
  </tr>
  <tr>
    <td>Convolution + RelU</td>
    <td>Field: 4x4, Stride: 1, Depth: 20</td>
    <td>28x28x20</td>
  </tr>
  <tr>
    <td>Max Pool</td>
    <td>Field: 2x2, Stride: 2</td>
    <td>14x14x20</td>
  </tr>
  <tr>
    <td>Convolution + RelU</td>
    <td>Field: 3x3, Stride: 1, Depth: 40</td>
    <td>12x12x40</td>
  </tr>
  <tr>
    <td>Convolution + RelU</td>
    <td>Field: 3x3, Stride: 1, Depth: 80</td>
    <td>10x10x80</td>
  </tr>
  <tr>
    <td>Max Pool</td>
    <td>Field: 2x2, Stride: 2</td>
    <td>5x5x80</td>
  </tr>
  <tr>
    <td>Convolution + RelU</td>
    <td>Field: 3x3, Stride: 1, Depth: 120</td>
    <td>3x3x120</td>
  </tr>
  <tr>
    <td>Convolution + RelU</td>
    <td>Field: 3x3, Stride: 1, Depth: 240</td>
    <td>1x1x240</td>
  </tr>
  <tr>
    <td>Fully Connected + RelU</td>
    <td>240x1000 Matrix</td>
    <td>1x1000</td>
  </tr>
  <tr>
    <td>Class Probabilities</td>
    <td>1000x3 Matrix</td>
    <td>1x3</td>
  </tr>
</table>
</center>
<br>
<br>

Now  we understand the constraint from input and output sizes and can confirm that our network satisfies it. We can also enumerate some other architectures satisfying this constraint, summarized in the list below (here 'F' stands for field size, 'S' stands for stride):

<center>
<table style  = "width:80%">
  <tr>
    <td><b>Network 1</b></td>
    <td><b>Network 2</b></td>
    <td><b>Network 3</b></td>
  </tr>
  <tr>
    <td>Convolution + RelU <br>(F:4x4, S:1)</td>
    <td>Convolution + RelU <br>(F:6x6, S:1)</td>
    <td>Convolution + RelU <br>(F:8x8, S:1)</td>
  </tr>
  <tr>
    <td>MaxPool <br>(F:2x2, S:2)</td>
    <td>MaxPool <br>(F:2x2, S:2)</td>
    <td>MaxPool <br>(F:2x2, S:2)</td>
  </tr>
  <tr>
    <td>Convolution + RelU <br>(F:5x5, S:1)</td>
    <td>Convolution + RelU <br>(F:4x4, S:1)</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
  </tr>
  <tr>
    <td>Convolution + RelU <br>(F:5x5, S:1)</td>
    <td>MaxPool <br>(F:2x2, S:2)</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
  </tr>
  <tr>
    <td>MaxPool <br>(F:2x2, S:2)</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
  </tr>
  <tr>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
    <td>MaxPool <br>(F:2x2, S:2)</td>
  </tr>
  <tr>
    <td>-</td>
    <td>-</td>
    <td>Convolution + RelU <br>(F:3x3, S:1)</td>
  </tr>
</table>
</center>
<br>
<br>

All these architectures satisfy the sizing constraint of our problem. Do they perform differently? Let's do a quick comparison of learning curves. I compare learning curves from the network we studied in our last post, and one of our alternative networks (Network 2 in the table above). The difference is clear. As we saw before our original network trains more quickly, reproducibly reaching the limit of improvement within our training. Network 2, on the other hand, learns more slowly and fails to reach this limit within our training time.

<div style="text-align:center">
<img src="assets/2017-09-25-learning-curves-comparison.png" />
<figcaption>Network architecture affects learning and performance. Original network (left) learns more quickly than alternative network 2 (right), achieving better performance more rapidly during training.</figcaption>
</div>
<br>
<br>

Different architectures thus learn differently - a behavior that isn't unexpected. In convolutional networks, there's often a tradeoff between a smaller number of convolutional layers each containing large kernels, or a larger number of convolutional layers that essentially stack smaller kernels. The second approach - stacking more smaller layers - is generally preferred. These two networks roughly make that tradeoff in the intermediate layer, where our original network stacks two layers of 3x3 kernels while the alternative architecture (Network 2) has a single 4x4 kernel layer. Additionally, Network 2 has fewer filters per layer. The end result is likely that Network 2 is just not as expressive as our original network. This means it ends up having a harder time fitting our data set, resulting in the slower training and higher error.

Many ML papers end up with a short section comparing architecture and design variations in their systems. Variations in deep learning papers include simple ones like the layer parameters considered here, completely different layer architectures including connections between non-adjacent layers, methods to minimize overfitting, and several others. Checking these features is important for network design, and can yield fun insights into a learning system's behavior.


<br>
<br>
<div style="text-align:center">
<img src="assets/2017-09-25-im-pei.jpg" />
<figcaption>This guy knows architecture matters.</figcaption>
</div>
