---
layout: post
title: Differentiating in Tensorflow 2.0
---

This will be a post about differentiating in Tensorflow 2.0 with an example of a loss function consisting of a differential of the neural network.

In the begining, I was working with neural networks either writing them from scrath in NumPy or with Keras framework. At some point during my experiments concerning the subject of my master thesis, I wanted to compare the results acquired with my from-scratch implementation with one written in a matured framework. My first choice was obviously Keras. Unfortunately, despite finding some interesting answers on the Stackoverflow, I could not achieve with it one, relatively simple, thing - I could not define a loss function build with derivative of the network. Here, in this post, I am going to show you how to use new Tensorflow 2.x object called Tape and how can it be used to differentiate anything and anywhere, e.g. network response in the loss function.

## Tensorflow 2.0 - new way to go

Let's start with the problem of defining a simple model in Tensorflow 2.x what can be found in [the official tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner).
{% highlight python %}
import tensorflow as tf
import numpy as np

X = tf.as_tensor(np.arange(0.1,2,0.1))
Y = tf.log(X)


model = tf.keras.models.Sequential([
    tf.keras.Layers.Dense(10, input_shape=(1,1),
                            activation='sigmoid'),
    tf.keras.Layers.Dense(1)
])

model.compile(optimizer='SGD', loss='mse', metrics=['mae'])

model.fit(X, Y)

{% endhighlight %}