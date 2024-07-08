import numpy as np
import numpy.random as rd

class Network(object):

    def __init__(self,size,function):
        """
        Parameters:
        size:list describing the number of neurons per layer ([3,4,1] 3 neurons layer 1, 4 neurons layer 2 and 1 neurons layer 3
        function:list descirbing the activation function to apply on the hidden layers and output layers ("reLu","tanh","sigmoid")
        """
        self.number_layer=len(size)
        self.size=size
        self.function=function
        self.bias=[rd.uniform(low=-1,high=1,size=(num_neurons,1)) for num_neurons in size[1:]] 
        self.weight=[rd.uniform(low=-1,high=1,size=(size[i+1],size[i])) for i in range(self.number_layer-1)]
        
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))
    
    def tanh(self,z):
        return(2*self.sigmoid(z)-1)
    
    def activation_apply(self,z,fun):
        """
        Parameters:
        z: float/list of float where to evaluate the function
        fun: string describing the type of function to choose

        Goal:
        Apply a chosen function.
        """
        if(fun=="reLu"):
            return(np.maximum(0,z))
        if(fun=="sigmoid"):
            return(self.sigmoid(z))
        if(fun=="tanh"):
            return(self.tanh(z))
        else:
            print("Unknow function, choices are:  \n reLu, sigmoid, tanh")
            raise SystemExit(0)

    def der_apply(self,z,fun):
        """
        Parameters:
        Same as before

        Goal:
        Similar to the function before, but for the derivative
        """
        if(fun=="reLu"):
            return(np.where(z>0,1,0))
        if(fun=="sigmoid"):
            return(self.sigmoid(z)*(1-self.sigmoid(z)))
        if(fun=="tanh"):
            return(1-np.power(self.tanh(z),2))

    def feedforward(self,input):
        """
        Parameters:
        Input: list of the data, look at the example for the dimension
        
        Goal:
        For each layer except the input one we compute the signal the neurons
        as the sum of the before layer signals times the weights of the concerned neuron 
        plus the bias of the concerned neurons and store these values
        """
        signal_layers=[np.array(input)]
        activation_layers=[np.array(input)]
        for i in range(self.number_layer-1):
            signal_layers.append(np.dot(self.weight[i],activation_layers[i])+self.bias[i])
            activation_layers.append(self.activation_apply((signal_layers[i+1]),self.function[i]))
        return(signal_layers,activation_layers)

    
    def fit(self,input,output,learning_rate,epoch):
        """
        Parameters:
        input: same as before
        output: list of the expected output associated to the input look at the example for the dimension
        learning_rate: float parameters of the gradient descent
        epoch: int how many time the fit function will pass through the data

        Goal:
        Train the net for a certain number of epoch
        """
        for _ in range(epoch):
            for i in range(len(input)):
                signal_layers,activation_layers=self.feedforward(input[i])
                delta=[np.subtract(activation_layers[-1],output[i])*self.der_apply(signal_layers[-1],self.function[-1])]
                for e in range(self.number_layer-2):
                    delta.insert(0,np.dot(self.weight[-1-e].T,delta[0])*self.der_apply(signal_layers[-2-e],self.function[-2-e]))
                for j in range(self.number_layer-1):
                    self.weight[j]-=learning_rate*np.dot(delta[j],activation_layers[j].T)
                    self.bias[j]-=learning_rate*delta[j]

    def pred(self,input):
        """
        Parameters:
        Same as before

        Goal:
        Once the net has been trained predict the result for a certain input
        """
        return(self.feedforward(input)[1][-1])
