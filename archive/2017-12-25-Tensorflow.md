---
layout: post
title: Tensorflow
comments : true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<link rel="stylesheet" href="./../css/prism.css">
<script src="./../js/prism.js"></script>

Having used Tensorflow in most of my projects, learning from the mistakes committed, I would like to share some of the mistakes that one might commit while building graphs in Tensorflow. At first glance these might look intuitive but closer inspection reveals a fatal mistake.

This post is related to some of the common mistakes that one might commit while building graphs in Tensorflow. First thing to keep in mind is that Tensorflow builds computational Graphs at compile time.

## Example 1

Using numpy in generating numbers. This could occur in the following cases

- Generating mask for a tensor (Incase of implementation of _**Dropout**_)
- Generating random numbers during run time

One might be tempted to do the following:

<pre>
  <code class="language-python">
   import tensorflow as tf
   import numpy as np

   input_val = tf.placeholder(tf.float32, shape=[5], name='input')
   tensor_mask = np.random.randint(low=0, high=5, size=5).astype(np.float32)
   result = tf.multiply(ran, input_val)

   # Run the graph
   sess = tf.Session()
   inp = np.array([1,2,3,4,5])
   init = tf.global_variables_initializer()
   sess.run(init)
   for i in xrange(10):
       print 'Idx: {:2d}, Random Vector: {},
              Output: {}'.format(i+1, ran, sess.run(result, feed_dict={input_val: inp}))
   sess.close()
  </code>
</pre>

we get the following output:
<pre>
   <code class="language-python">
   Idx:  1, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  2, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  3, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  4, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  5, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  6, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  7, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  8, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx:  9, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   Idx: 10, Random Vector: [ 0.  1.  4.  3.  4.], Output: [  0.   2.  12.  12.  20.]
   </code>
</pre>>

Hmmm.... This is not something we expected to occur!! The problem is because the mask is generated at the compile time and the variable **tensor_mask** is fixed. How do we correct this? This can be correctly implemented as follows:

<pre>
   <code class="language-python">
   input_val = tf.placeholder(tf.float32, shape=[5], name='input')
   tensor_mask = tf.random_uniform(minval=0, maxval=5, shape=[5], dtype=tf.float32)
   result = tf.multiply(tensor_mask, input_val)

   # Run the graph
   sess = tf.Session()
   inp = np.array([1,2,3,4,5])
   init = tf.global_variables_initializer()
   sess.run(init)
   for i in xrange(10):
       r, f = sess.run([tensor_mask, result], feed_dict={input_val: inp})
       print 'Idx: {:2d}, Random Vector: {}, Output: {}'.format(i+1, r, f)

   </code>
</pre>>

Running the above code, produces the following result:
<pre>
   <code class="language-python">
   Idx:  1, Random Vector: [ 3.25267553  4.84795475  1.4029181   3.10610414  4.83795547], Output: [  3.25267553   9.6959095    4.20875454  12.42441654  24.18977737]
   Idx:  2, Random Vector: [ 3.01925611  1.97615147  4.0551405   0.23586869  0.15114605], Output: [  3.01925611   3.95230293  12.16542149   0.94347477   0.75573027]
   Idx:  3, Random Vector: [ 1.51890755  1.09132946  1.00828886  4.03317404  0.42661846], Output: [  1.51890755   2.18265891   3.02486658  16.13269615   2.1330924 ]
   Idx:  4, Random Vector: [ 3.82456255  3.5389204   4.74261808  3.85795641  0.34844756], Output: [  3.82456255   7.07784081  14.22785378  15.43182564   1.74223781]
   Idx:  5, Random Vector: [ 1.66118205  0.51932991  4.01084948  0.4434216   2.05401945], Output: [  1.66118205   1.03865981  12.0325489    1.77368641  10.27009773]
   Idx:  6, Random Vector: [ 1.79597974  3.93435836  0.3850919   0.83782196  0.79356313], Output: [ 1.79597974  7.86871672  1.1552757   3.35128784  3.96781564]
   Idx:  7, Random Vector: [ 1.16980672  1.14684701  2.32673216  4.42601299  0.6166929 ], Output: [  1.16980672   2.29369402   6.98019648  17.70405197   3.08346462]
   Idx:  8, Random Vector: [ 0.45332193  2.15635538  0.14178693  1.8891263   2.6841116 ], Output: [  0.45332193   4.31271076   0.4253608    7.5565052   13.42055798]
   Idx:  9, Random Vector: [ 3.44323874  0.07824361  1.56545639  4.30905914  0.18945515], Output: [  3.44323874   0.15648723   4.69636917  17.23623657   0.94727576]
   Idx: 10, Random Vector: [ 4.70608711  1.22823417  4.27506876  0.98638773  1.29608154], Output: [  4.70608711   2.45646834  12.82520676   3.94555092   6.48040771]
   </code>
</pre>>

which is the way we want it to work !


## Example 2

Implementing Batch Normalization. Most important thing to keep in mind is the way we share the batch statistics during train and test time. The way batch norm works briefly is as follows:

_
_