import Perceptron as p

print("\nThis is an example of a neural network learning how to implement a XOR gate")

#test code
mlp = p.MultiLayerPerceptron(layers=[2,2,1])
print("\nTraining Neural Network for 3k epoches...\n")

for i in range(3000):
    MSE = 0.0
    MSE += mlp.bp([0,0],[0])
    MSE += mlp.bp([0,1],[1])
    MSE += mlp.bp([1,0],[1])
    MSE += mlp.bp([1,1],[0])
    MSE = MSE / 4
    if(i%100 == 0):
        print (MSE)

mlp.printWeights()
    
print("Multi Layer Perceptron:")
print ("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))


