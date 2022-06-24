import numpy as np
import sys

class Perceptron: #typology of neuron with:
	# n input sample [0 - n-1]
	# 1 bias point (offset for the neural decision threshold)
	# 1 neuron able to sum the weigthed input
	# 1 output non linear function to betterer distinguish between data (sigmoid activation function)


	#Attributes: input number, bias term (default 1.0)

	def __init__(self,inputs, bias=1.0):
		#create a perceptron object with random weight of inputs
		self.weights = (np.random.rand(inputs+1)*2) -1 
		self.bias = bias
		self.n_input = inputs
		
	def run(self, x): 
		# x is the list of inputs
		sum = np.dot(np.append(x, self.bias),self.weights) # vectorial product: (input vector and bias) * (weights)
		
		return self.sigmoid(sum) #the weighted inputs are passed to the activation function
		
	def set_weights(self, w_init):
		#specify the values of the weights, they cover also the bias
		if len(w_init) == (self.n_input + 1) :  
			self.weights = np.array(w_init)	

		else:
			print("Error: A neuron has received a worng number of input", sys.stderr)
			
			if len(w_init) < (self.n_input + 1) :  
				base=np.zeros(self.n_input + 1)
				self.weigths = np.array(w_init+base)	
			else:   
				self.weigths = np.array(w_init[:(self.n_input + 1)])	
			
	
	def sigmoid(self, x):
		# normalizing the output of the neuron with the sigmoid function
		return 1/(1+np.exp(-x))

	
	

class MultiLayerPerceptron:      
	#A multilayer perceptron class that uses the Perceptron class above.   
	#Attributes:
        #  layers:  A python list with elements the number of neurons per layer.
        #  bias:    The bias term. The same bias is used for all neurons.
        #  eta:     The learning rate.'''

    def __init__(self, layers, bias = 1.0):
        #Return a new MLP object with the specified parameters.
        self.layers = np.array(layers,dtype=object)
        self.bias = bias	#same for all the network, the weights can change
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)

    def set_weights(self, w_init):
    	#initilize the weights of a network of an arbitrary number of elements
    	#w_init is a list of lists, is a list of neuron where each neuron is define by a sequence (list) of its weights.
    	#for the moment it iterates only on the various neuros in the various layer
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])
                #set_weights is called again but now to inizialize each single neuron with its weights
                #It start from the layer i+1 becouse network[0] is input layer (= no neuron there)


    def printWeights(self):
    	#Let the user see the weights of the networks, useful when training
        print("\n")
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print("\n")

    def run(self, x): #Feed a sample x into the MultiLayer Perceptron.
    	
    	#x is trnansported to an array, then is associated to the first network layer, the input
        x = np.array(x,dtype=object)
        self.values[0] = x
        #Here are called all the neurons one after the other 
        #From the first layer, all the neuron of this layer are called one after the other, then for the second layer, and so on
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):  
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
                
        return self.values[-1] #array of the output layer of the network













