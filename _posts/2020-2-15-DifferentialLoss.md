---
layout: post
title: Differentiating in Tensorflow 2.0
---

Tensorflow 2.x brings a totally different quality of life embracing the declarative differential
programming style of coding. In this post, I show how easily can this new approach be used to
differentiate any tensor at any place in the code.

In the beginning, I was working with neural networks either writing them from scratch in NumPy
or with Keras. At some point during the experiments concerning the subject of my master thesis,
I wanted to compare the results acquired with my from-scratch implementation with one written
in a matured framework. My first choice was obviously Keras. Unfortunately, despite finding some
interesting answers on the Stackoverflow, I could not achieve with it one, relatively simple,
thing - I could not define a loss function with a derivative of the network. Here, in this post,
I am going to show you how to use new Tensorflow 2.x object called Tape and how can it be used to
differentiate anything and anywhere, e.g. network response in the loss function.

## Tensorflow 2.x - the new way to go

Let's start with the problem of defining a simple model in Tensorflow 2.x what can be found also
in [the official tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner).
It adapts Keras as a high level abstraction API what allows us to define a model in just a couple
of lines. Here, we are going to fit a network to approximate a logarithm function as
in [the acompanying Colab notebook](https://colab.research.google.com/drive/17P6XDmYbNULQOgnEjxJJDwOYfaNbxopK).

We begin with standard imports and the definition of our samples **X** and targets **Y** sets.

{% highlight python %}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0.1,2,0.1)
Y = np.log(X)
{% endhighlight %}

Now, we define a model using the Sequential class that represents a simple feedforward network.
In this case, the network will be constructed from three densely connected layers.
The first two are composed of 10 units and end with sigmoid activation function.
The third one has a single linear output. We should specify the input shape.
It is not necessary, but it automatically calls **build()** method what allows us to summarize our model.
{% highlight python %}
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,),
                            activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])
model.summary()
{% endhighlight %}

By calling the **summary()** method, we receive the following description of the model.
![Summary of the model]({{ site.baseurl }}/images/diffinTF2/summary.jpg "Summary")

With the method **compile()**, we specify the optimizer (in this case stochastic gradient descent),
a loss function (mean square error), and a list of metrics (mean absolute error).
{% highlight python %}
model.compile(optimizer=tf.keras.optimizers.SGD(
    learning_rate=0.1),
    loss='mse',
    metrics=['mae'])
{% endhighlight %}

Finally, we can train the network with the method **fit()**. The **verbose=0** parameter is specified for no direct information.
{% highlight python %}
history = model.fit(x=X, y=Y, epochs=10000, verbose=0)
{% endhighlight %}

By turning on the information with **verbose=1**, we get the following message.
![Training message]({{site.baseurl}}/images/diffinTF2/fit.jpg "Training message")

We can see if the network learned the correct solution.
{% highlight python %}
plt.title("Fitting a network to a logarithm")
plt.plot(X, Y, label='Ground truth')
plt.plot(X, model.predict(X), label='Prediction')
plt.legend()
plt.show()
{% endhighlight %}
![Logarithm approximation]({{site.baseurl}}/images/diffinTF2/function.jpg "Approximated function")

Thanks to the **history** object that is returned by the **fit()** method,
we can check how did the loss and metrics change during the learning process.
{% highlight python %}
plt.title('Loss function - MSE')
plt.plot(np.arange(1, len(history.history['loss'])+1),
        history.history['loss'])
plt.xlabel('Epochs')
plt.show()
{% endhighlight %}
![Loss function]({{site.baseurl}}/images/diffinTF2/loss.jpg "Loss function - MSE")

{% highlight python %}
plt.title('Metric - MAE')
plt.plot(np.arange(1, len(history.history['mean_absolute_error'])+1),
        history.history['mean_absolute_error'])
plt.xlabel('Epochs')
plt.show()
{% endhighlight %}
![Metric function]({{site.baseurl}}/images/diffinTF2/metric.jpg "Metric - MAE")

This sums up the basic tools we need for simple supervised tasks. If you encountered Keras before
as a standalone framework, you can see that Tensorflow 2.x incorporated the high level as it was.
This is a great news as this is a very intuitive and easy to use.
Now, we are going to do something what would be problematic with standard Keras,but TF2 can handle it.

## Differential loss function

Now, let's consider a similar problem to the previous one, but as the derivative of the
logarithmic function is known, we can change the target from the logarithm to $1/x$.
{% highlight python %}
Z = X**-1
{% endhighlight %}