---
title: Segmentation 4 - Fast and Fully Connected
permalink: /segmentation-fully-connected
status: published
---


Our previous posts glossed over a crucial practical detail. How did we actually segment images? The careful reader will note I punted that discussion to a later post. So here it is.

Segmenting images with convolutional networks presents a bit of a conundrum. Recall the method's main idea - classify each pixel using a small surrounding image patch to provide contextual information about its identity. This sounds great - until you think about classifying two adjacent pixels.

Two adjacent pixels share almost identical surrounding image patches. In our case, two adjacent 31x31 pixel patches overlap by almost 97%. Yet to classify both pixels, we ask our CNN to perform identical computations on both pixel-patches despite the overwhelming overlap, leading to loads of redundant computation.

This feels like a lot of wasted effort. There must be a better way.

And of course, there is. The solution people employ is the *fully convolutional network*. While a naive segmentation network classifies by processing individual pixel patches one at a time, a fully convolutional network processes the entire image in one shot without excessive redundant computation. It does this by arranging network's filters in a clever way to reuse computed information. And in fact, one can convert a standard convolutional network for segmentation into a fully convolutional network.

Several research papers discuss the fully convolutional concept. I'll discuss an implementation of a [preprint](https://arxiv.org/abs/1412.4526) by Li *et al* 2014, "Highly Efficient Forward and Backward Propagation of Convolutional Neural Networks for Pixelwise Classification." This preprint describes a very accessible method with which to convert a standard patch-wise convolutional segmentation network to a fully convolutional network. The method converts each operation layer of the standard network into its fully convolutional counterpart using a concept they call "d-regulary sparse kernels."
<br>
<div style="text-align:center">
<img src="assets/2017-09-27-colonel.jpg" width="400" height="300"/>
<figcaption>We're talking about kernels, not colonels.</figcaption>
</div>
<br>
What do Li *et al*'s d-regularly sparse kernels entail?

The basic idea is to first transform each operation in the original network into a 'sparse,' or 'dilated' version. An operator is dilated by inserting zero rows and columns between its original values, as in this figure from the preprint:

<div style="text-align:center">
<img src="assets/2017-09-27-dilation-matrix-diagram.png" width = "600" height = "500"/>
<figcaption>Dilating an operator by factor <i>d</i> inserts <i>d</i> zero rows and columns between the operator's original values. Top - original convolutional (a) and Max-Pool (b) kernels. Bottom - dilated kernels with dilation factor (c) 2 and (d) 3. (Source - Li *et al* 2014.)</figcaption>
</div>
<br>
<br>


Beginning at the network's input, we initially do not dilate operators at all. Once, however, we hit an operator with stride greater than 1, we dilate all subsequent operations by that stride. Each subsequent operator with stride greater than one contributes multiplicatively to the dilation. For example, all operators following a stride 3 convolution and then a stride 2 max pool would be dilated by 6. We end up creating a fully convolutional network using these dilated operators in place of their corresponding original operators.

For those interested in a visual summary of these details, I suggest consulting Figure 3 of the original [preprint](https://arxiv.org/abs/1412.4526). The figure is a little challenging to parse on first reading, but once you see it you'll grasp the mapping well.


We can readily implement this method using Tensorflow. Let's write out how to do this for the convolutional portions of our original network. The key option to use is `dilation_rate` in `tf.nn.convolution()` and `tf.nn.pool()`. This allows us to directly implement the d-regularly sparse kernels of Li *et al*.

```
d = 1

h1_conv = tf.nn.convolution(x, W1, padding = 'VALID', strides = [1,1], dilation_rate = [d,d]) + b1
h1_relu = tf.nn.relu(h1_conv)
h1_pool = tf.nn.pool(h1_relu, window_shape = [2,2], pooling_type = 'MAX', padding = 'VALID', dilation_rate = [d,d], strides = [1,1])

d = d * 2 # h1_pool originally had stride of 2

h2_conv = tf.nn.convolution(h1_pool, W2, padding = 'VALID', strides = [1,1], dilation_rate = [d,d]) + b2
h2_relu = tf.nn.relu(h2_conv)
h3_conv = tf.nn.convolution(h2_relu, W3, padding = 'VALID', strides = [1,1], dilation_rate = [d,d]) + b3
h3_relu = tf.nn.relu(h3_conv)
h3_pool = tf.nn.pool(h3_relu, window_shape = [2,2], pooling_type = 'MAX', padding = 'VALID', dilation_rate = [d,d], strides = [1,1])

d = d * 2 # h3_pool originally had stride of 2

h4_conv = tf.nn.convolution(h3_pool, W4, padding = 'VALID', strides = [1,1], dilation_rate = [d,d]) + b4
h4_relu = tf.nn.relu(h4_conv)
h5_conv = tf.nn.convolution(h4_relu, W5, padding = 'VALID', strides = [1,1], dilation_rate = [d,d]) + b5
h5_relu = tf.nn.relu(h5_conv)
```

The rest of the code organization (loading data, setting up variables and placeholders, running a Tensorflow session) follows the structure described in the second post. It's pretty standard, and I omit it here for brevity. One may need to do some fiddling with memory - on my system I needed to break the segmented image up into four pieces to fit the entire operation in memory.

How fast is this code? On an Amazon `g2.xlarge` instance this algorithm segments a 1040x1392 image in 3.5 seconds. The naive patch-by-patch approach takes 48 minutes.

That's an 822x speed-up.

In other words, a game changer.
