#!/bin/python3

import numpy as np
from Perceptron import *

EXERCISE=1

#test code
neuron = Perceptron(inputs=2)

#The capability of prediction is determined by the weight of the neurons

if EXERCISE==1:
	#Here the weight are assigned in order to obtain an AND logic gate behaviour:
	neuron.set_weights([10,10,-15])  #AND

	print("\nAND gate:\n")
	print ("0 0 = {0:.5f}".format(neuron.run([0,0])))
	print ("0 1 = {0:.5f}".format(neuron.run([0,1])))
	print ("1 0 = {0:.5f}".format(neuron.run([1,0])))
	print ("1 1 = {0:.5f}".format(neuron.run([1,1])))


elif EXERCISE==2:
	#Here the weight are assigned in order to obtain an OR logic gate behaviour:
	neuron.set_weights([10,10,-5])  #OR

	print("\nOR gate:\n")
	print ("0 0 = {0:.5f}".format(neuron.run([0,0])))
	print ("0 1 = {0:.5f}".format(neuron.run([0,1])))
	print ("1 0 = {0:.5f}".format(neuron.run([1,0])))
	print ("1 1 = {0:.5f}".format(neuron.run([1,1])))


elif EXERCISE==3:
	#This is a multi-layer Perceptron, realized in a educational, clear, way
	
	#An EXOR can't be realized by only one Perceptron! the trasfer function doesn't allow a single linear separation between the cathegories
	#Three Perceptron are neeeded, the arrengement rassemble the classic IC implementation of an EXOR:
	#EXOR = AND(NAND,OR)

	print("\nDescrete example of XOR gate:\n")

	neuron_nand = Perceptron(inputs=2)	 # AND * -1
	neuron_nand.set_weights(np.array([10,10,-15])*-1)  # NAND ( [...]*-1 doesn't work )

	neuron_or = Perceptron(inputs=2)
	neuron_or.set_weights([10,10,-5])  # OR

	neuron_and = Perceptron(inputs=2)
	neuron_and.set_weights([10,10,-15])	   # AND
	#NB the bias normally are all 1.0 and change only the weights 

	A, B = 0, 0
	Y1 = neuron_nand.run([A, B])
	Y2 = neuron_or.run([A, B])
	print ("0 0 = {0:.5f}".format( neuron_and.run([Y1, Y2]) ) )
	A, B = 1, 0
	Y1 = neuron_nand.run([A, B])
	Y2 = neuron_or.run([A, B])
	print ("1 0 = {0:.5f}".format( neuron_and.run([Y1, Y2]) ) )
	A, B = 0, 1
	Y1 = neuron_nand.run([A, B])
	Y2 = neuron_or.run([A, B])
	print ("0 1 = {0:.5f}".format( neuron_and.run([Y1, Y2]) ) )
	A, B = 1, 1
	Y1 = neuron_nand.run([A, B])
	Y2 = neuron_or.run([A, B])
	print ("1 1 = {0:.5f}".format( neuron_and.run([Y1, Y2]) ) )
	

elif EXERCISE==4:

	print("\nMulty-layer implementation of XOR gate:\n")
	mlp = MultiLayerPerceptron(layers=[2,2,1])  #mlp
	mlp.set_weights([[[-10,-10,15],[10,10,-5]],[[10,10,-15]]])
	mlp.printWeights() #sanity test

	print ("0 0 = {0:.5f}".format(mlp.run([0,0])[0]))
	print ("0 1 = {0:.5f}".format(mlp.run([0,1])[0]))
	print ("1 0 = {0:.5f}".format(mlp.run([1,0])[0]))
	print ("1 1 = {0:.5f}".format(mlp.run([1,1])[0]))


