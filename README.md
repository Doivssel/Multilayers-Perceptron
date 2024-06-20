# Perceptron

What I present here is a small package that may be used to create multilayers perceptrons. Before explaining how to use this package let's look at the math behind to understand how to construct a perceptron.

Let's denote N the number of layers and $U_1,.....,U_N$ the number of neurons per layer (ex: $U_1$ is the number of neurons on the first layer)

Now we define a list for the bias  as follow $\mathbf{B} =$ { $\mathbf{b}^{(2)}, \mathbf{b}^{(3)}, \ldots, \mathbf{b}^{(N)} $} with
```math 
\mathbf{b}^{(2)} = \begin{bmatrix}b^{(2)}_1 \\b^{(2)}_2 \\\vdots \\b^{(2)}_{U_1}\end{bmatrix}
 ,............,
\mathbf{b}^{(N)} = \begin{bmatrix}b^{(N)}_1 \\b^{(N)}_2 \\\vdots \\b^{(N)}_{U_N}\end{bmatrix}
```
we do not take the first layer into account as it is the input layer, as such it has no need for bias. As an example the $b^{(2)}_1$ is the bias on the first neurons of the second layer.

And the weight matrix

We then can compute the propagated signals on each layers 
