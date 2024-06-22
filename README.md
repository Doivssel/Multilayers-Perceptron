# Perceptron

What I present here is a small package that may be used to create multilayers perceptrons. Even if multilayers perceptron are not that useful due to their training time and the number of hyperparameters that cannot be learned (number of neurons in hidden layers, number of layers, learning rate...). But this is still interesting to learn as it is the basic of neural networks.

So let's get into it, here is a quick explanation of the math behind it. Let's denote N the number of layers and $U_1,.....,U_N$ the number of neurons per layer (ex: $U_1$ is the number of neurons on the input layer)

Now we define a list for the bias matrices as follow $\mathbf{B} =$ { $\mathbf{b}^{(2)}, \mathbf{b}^{(3)}, \ldots, \mathbf{b}^{(N)} $} with
```math 
\mathbf{b}^{(2)} = \begin{bmatrix}b^{(2)}_1 \\b^{(2)}_2 \\\vdots \\b^{(2)}_{U_1}\end{bmatrix}
 ,............,
\mathbf{b}^{(N)} = \begin{bmatrix}b^{(N)}_1 \\b^{(N)}_2 \\\vdots \\b^{(N)}_{U_N}\end{bmatrix}
```
we do not take the first layer into account as it is the input layer, as such it has no need for bias. As an example the $b^{(2)}_1$ is the bias on the first neurons of the second layer.

And a list for the weights matrices $\mathbf{W} =$ { $\mathbf{W}^{(2)}, \mathbf{W}^{(3)}, \ldots, \mathbf{W}^{(N)}$ }.
```math
\mathbf{W}^{(2)} = \begin{bmatrix}w^{(2)}_{1,1} & w^{(2)}_{1,2} & \cdots & w^{(2)}_{1,U_1} \\w^{(2)}_{2,1} & w^{(2)}_{2,2} & \cdots & w^{(2)}_{2,U_1} \\\vdots & \vdots & \ddots & \vdots \\w^{(2)}_{U_2,1} & w^{(2)}_{U_2,2} & \cdots & w^{(2)}_{U_2,U_1}\end{bmatrix}
 ,............,
\mathbf{W}^{(N)} = \begin{bmatrix}w^{(N)}_{1,1} & w^{(N)}_{1,2} & \cdots & w^{(N)}_{1,U_{N-1}} \\w^{(N)}_{2,1} & w^{(N)}_{2,2} & \cdots & w^{(N)}_{2,U_{N-1}} \\\vdots & \vdots & \ddots & \vdots \\w^{(N)}_{U_N,1} & w^{(N)}_{U_N,2} & \cdots & w^{(N)}_{U_N,U_{N-1}}\end{bmatrix}
```

We then can compute the propagated signals on each layers, let's denote $A^{(1)}=$ { $\mathbf{a_1^{(1)}},\mathbf{a_2^{(1)}},....,\mathbf{a_{U_1}^{(1)}}$ } the input signals of the input layers, we can then define recursively the signals for the others layers as



