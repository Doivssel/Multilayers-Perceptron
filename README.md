# Multilayers Perceptron

## Introduction

What I present here is a small package that may be used to create multilayers perceptrons. This is mainly for learning purpose, if you want to implement one it may be better to use some of the well known library such as Keras, Pytorch or Scikit.

So what's a multilayer perceptron ? It is a neural network with N layers, the first one being the input layer and the last one being the output layer. Each layer is composed of a certain number of neurons connected to all the other neurons of the next layer if there is any. Each of these connections has a weight that translate the importance of the connection between two neurons. The neurons themselves are a computing unit where they receive signals of the before layer multiplied by the weight of the concerned neurons plus the bias of the neuron receiving signals. The neurons then send off theses signals after passing them through an activation function (sigmoid,tanh,reLu...) to the next layer, until the output layer is reached. In the end the goal is to put an input signal and let the perceptron compute the right answer. To do that it learn through multiple inputs the right biases and weights through an algorithm called gradient descent. Whose goal is to minimise the error between the output of our perceptron and the expected value. As such this version of the perceptron fall under the supervised learning category.

## Initialisation

So let's get into it, here is a quick explanation of the math behind it. Let's denote N the number of layers and $U_1,.....,U_N$ the number of neurons per layer (ex: $U_1$ is the number of neurons on the input layer)

Now we define a list for the bias matrices as follow $\mathbf{B} =$ { $\mathbf{b}^{(2)}, \mathbf{b}^{(3)}, \ldots, \mathbf{b}^{(N)} $} with
```math 
\mathbf{b}^{(2)} = \begin{bmatrix}b^{(2)}_1 \\b^{(2)}_2 \\\vdots \\b^{(2)}_{U_2}\end{bmatrix}
 ,............,
\mathbf{b}^{(N)} = \begin{bmatrix}b^{(N)}_1 \\b^{(N)}_2 \\\vdots \\b^{(N)}_{U_N}\end{bmatrix}
```
we do not take the first layer into account as it is the input layer, as such it has no need for bias. As an example the $b^{(2)}_1$ is the bias on the first neurons of the second layer.

And we also define a list for the weights matrices $\mathbf{W} =$ { $\mathbf{W}^{(2)}, \mathbf{W}^{(3)}, \ldots, \mathbf{W}^{(N)}$ }.
```math
\mathbf{W}^{(2)} = \begin{bmatrix}w^{(2)}_{1,1} & w^{(2)}_{1,2} & \cdots & w^{(2)}_{1,U_1} \\w^{(2)}_{2,1} & w^{(2)}_{2,2} & \cdots & w^{(2)}_{2,U_1} \\\vdots & \vdots & \ddots & \vdots \\w^{(2)}_{U_2,1} & w^{(2)}_{U_2,2} & \cdots & w^{(2)}_{U_2,U_1}\end{bmatrix}
 ,............,
\mathbf{W}^{(N)} = \begin{bmatrix}w^{(N)}_{1,1} & w^{(N)}_{1,2} & \cdots & w^{(N)}_{1,U_{N-1}} \\w^{(N)}_{2,1} & w^{(N)}_{2,2} & \cdots & w^{(N)}_{2,U_{N-1}} \\\vdots & \vdots & \ddots & \vdots \\w^{(N)}_{U_N,1} & w^{(N)}_{U_N,2} & \cdots & w^{(N)}_{U_N,U_{N-1}}\end{bmatrix}
```
## Forward propagation

We then can compute the propagated signals on each layers, we have the following input signals 
```math 
\mathbf{A^{(1)}}=\begin{bmatrix}a^{(1)}_1 \\a^{(1)}_2 \\\vdots \\a^{(1)}_{U_1}\end{bmatrix}
```
from this we can get the signals of our second layers by computing $\mathbf{Z^{(2)}}=\mathbf{W}^{(2)}\mathbf{A^{(1)}}$ and by applying the activation function of the second layer $\sigma^{(2)}$ to $\mathbf{Z^{(2)}}$ as such we have $\mathbf{A^{(2)}}=\sigma^{(2)}(\mathbf{Z^{(2)}})$. From this it easy to see that we can define define recursively the signals for the l layer as  $\mathbf{A^{(l)}}=\sigma^{(l)}(\mathbf{Z^{(l)}})$ with $\mathbf{Z^{(l)}}=\mathbf{W}^{(l)}\mathbf{A^{(l-1)}}$

## Backward propagation

Now we want our model to learn the weight and bias. For that we use backward propagation, an algorithm that send the error of the output layer backward. Let's denote the loss as $\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}})$, where $\mathbf{\hat{y}}$ is the predicted output and $\mathbf{y}=A^{(N)}$ is the actual output.

We begin by computing the loss of the output layer by computing $\delta^{(N)} = \nabla_{\mathbf{A}^{(N)}} \mathcal{L} \odot \sigma^{(N)}{'}(\mathbf{Z}^{(N)})$. 

We then compute recursively the error by going backward as $\delta^{(l)} = (\mathbf{W}^{(l+1)})^T \delta^{(l+1)} \odot \sigma^{(l)}{'}(\mathbf{z}^{(l)})$ (we do not compute it for the input layer). Where $\odot$ is the Hadamard product and $\sigma^{(l)}{'}$ is the derivative the activation function of the l layer.

## Gradient descent

Now you may wonder why we've computed this. It's to use the gradient descent algorithm to find the biases and weights that minimise the error. To do that gradient descent move in the direction where of the steepest descent by moving in the opposite direction of the gradient. It is important to note that this work well only when the loss function is convex. In my code I implent a simple sum of squared.

In other word to find the optimum we must compute the gradient of the loss, as it may be seen as a function of the bias and weight. As such we can obtain the following rule to update our 
parameters,

$$\mathbf{W}^{(l)} = \mathbf{W}^{(l)} - \eta \nabla_{\mathbf{W}^{(l)}}$$

$$\mathbf{b}^{(l)} = \mathbf{b}^{(l)} - \eta \nabla_{\mathbf{b}^{(l)}}$$

with $\nabla_{\mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$ and $\nabla_{\mathbf{b}^{(l)}} = \delta^{(l)}$. Most of these results can be obtained by taking the partial derivative of the loss under the biases/weights and then using chain rule.

## Shortcoming

Multilayers perceptron work well enough for some easy task such as solving the Xor problem. But for more difficult problem we quickly encouter many difficulties mainly due to the the parameters that cannot be learned such as the learning rate, the number of layer, the number of neurons or on how to initialise the biases and weights. Even the slighest change may make our neural network useless. It may be interesting to use other model for more difficult tasks. 

## A simple example

I've given a simple example on the famous problem of the XOR. The neural net trained was able to perform perfectly on this simple problem.

## Code

Concerning the code there is not much to say I've implemented the above parts. To use it just look at the code I've commented it quite well.


