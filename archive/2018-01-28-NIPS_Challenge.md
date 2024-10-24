---
layout: post
title: NIPS'17 Paper Implementation challenge
comments: true
---


<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<link rel="stylesheet" href="./../css/prism.css">
<script src="./../js/prism.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


This blog is accompanied by the code implementation [here](https://github.com/kvmanohar22/img2imgGAN)

Having some time to kill during my winter vacation, I decided to implement a NIPS 2017 paper as part of NIPS Paper Implementation Challenge. At that moment I was very excited about GANs and their applications. Browsing through the papers, I finally settled on imlementing the paper,

_**Toward Multimodal Image-to-Image Translation** \\
 Jun-Yan Zhu et. al_

This paper, as the title suggests dwells on the problem of translating images from one domain to another domain. For eg: **edges2shoes**, **edges2handbags**, **day2night**, **labels2facades** and so on. So how is this different from other proposals such as _pix2pix_ ? And again the title in itself speaks of **Multimodality**. _**pix2pix**_ had a sort of one to one mapping. By one to one mapping I mean to say, a particular say edges layout of an object corresponds to only one output. This need not be the case. For eg: an outline of a shoe might correspond to different colors or color of a flower might be yellow, red, orange etc or we could have a layout of a building that could correspond to different colors or in case of colorization of black and white images, one particular grayscale might correspond to various colors. This is exactly the problem that is addressed in the paper with innovative model architectures to enforce the notion of multimodality in the structure of the generated image.


Being quite familiar with Tensorflow and for it's beautiful graph, summary visualizations using Tensorboard, I decided to use this framework. I find it very easy to debug the graph nodes connections with visualizations and in fact I could spot some of the most grieve mistakes and rectify them instantaneously. With the correct name/variable scoping one can generate graphs that can be very useful. Following is the graph generated for **cLR-GAN** model.

<br>
<br>

![](../images/nips/clr_gan.gif)
**Fig 1** Graph showing the structure of cLR-GAN (conditional Latent Regressor GAN) model.

I would like to outline here the problems that I faced and the steps that I took to overcome them.

## **Image Normalization**
I normalized the input images to $$[0, 1]$$ but squashed the output of generator network to $$[-1, 1]$$ using **tanh** non-linearity. oops !

## **Batch Normalization**
Implementing this can really get tricky with so many implementations of this within tensorflow itself. I used the following implementation:

<pre>
  <code class="language-python">
   def batch_normalize(input, is_training, reuse=False, name=None):
      """Applies batch normalization

      Args:
         input      : Input Tensor
         is_training: Operation is in train / test time
         reuse      : Whether to reuse the batch statistics
         name       : Optional name for the operation

      Returns:
         Tensor after batch normalization operation
      """
      if name is None:
         name = "BatchNorm"

      with tf.variable_scope(name, reuse=reuse):
         output = tf.contrib.layers.batch_norm(inputs=input,
                                               center=True,
                                               scale=True,
                                               is_training=is_training,
                                               updates_collections=None,
                                               scope=name)
         return output
  </code>
</pre>

what's **very important** in the above implementation is setting the flags **is_training** and **updates_collections**.

- updates_collections 

During train mode, the tensors, `moving_mean` and `moving_variance` need to be updated. This can be accomplished in two ways. By default the operations responsible for updating these are placed in `tf.GraphKeys.UPDATE_OPS` collection, so we must add these operations to control_dependencies before executing batch_norm,

<pre>
  <code class="language-python">
   update_ops = tf.get_collections(tf.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
      output = tf.contrib.layers.batch_norm(inputs=input,
                                            center=True,
                                            scale=True,
                                            is_training=is_training,
                                            scope=name)
  </code>
</pre>

The above line `with tf.control_dependencies(update_ops)` executes the operations in the parameter `update_ops` before executing any of the operations under this scope.

The other easy way around is to set `updates_collections=None` which updates the variables in place. But as mentioned in tensorflow docs,

> One can set updates_collections=None to force the updates in place, but that
> can have a speed penalty, especially in distributed settings.

- is_training

Set this to `True` during train mode and `False` during test/val phase.

<div class="fig figcenter fighighlight">

  <img src="../images/nips/result/g_0.png" width="23%" style="margin-right:3px;"> 
  <img src="../images/nips/result/t_0.png" width="23%" style="margin-right:2px;">
  <img src="../images/nips/result/cLR_0.png" width="23%" style="margin-right:3px;">
  <img src="../images/nips/result/cVAE_0.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="../images/nips/result/g_1.png" width="23%" style="margin-right:3px;"> 
  <img src="../images/nips/result/t_1.png" width="23%" style="margin-right:2px;">
  <img src="../images/nips/result/cLR_1.png" width="23%" style="margin-right:3px;">
  <img src="../images/nips/result/cVAE_1.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="../images/nips/result/g_2.png" width="23%" style="margin-right:3px;"> 
  <img src="../images/nips/result/t_2.png" width="23%" style="margin-right:2px;">
  <img src="../images/nips/result/cLR_2.png" width="23%" style="margin-right:3px;">
  <img src="../images/nips/result/cVAE_2.png" width="23%">

  <div class="figcaption">
    <b>Result</b> First column represents input, second column the ground truth. The next is the image generated from cLR-GAN and the last column represents the image generated from cVAE-GAN
  </div>
</div>


<!--<div class="fig figcenter fighighlight">
  <img src="" width="23%" style="margin-right:3px;"> 
  <img src="" width="23%" style="margin-right:2px;">
  <img src="" width="23%" style="margin-right:3px;">
  <img src="" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="" width="23%" style="margin-right:3px;"> 
  <img src="" width="23%" style="margin-right:2px;">
  <img src="" width="23%" style="margin-right:3px;">
  <img src="" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="" width="23%" style="margin-right:3px;"> 
  <img src="" width="23%" style="margin-right:2px;">
  <img src="" width="23%" style="margin-right:3px;">
  <img src="" width="23%">
</div>
 -->



<br>
<div id="disqus_thread"></div>
<script>
(function() {
var d = document, s = d.createElement('script');
s.src = 'https://kvmanohar22-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
