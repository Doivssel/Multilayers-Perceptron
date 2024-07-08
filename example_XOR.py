from multilayers_perceptron import *

rd.seed(10000000) #for reproductibility


input_Xor=[[[0],[0]], #truth table of XOR
           [[0],[1]],
           [[1],[0]],
           [[1],[1]]]

output_Xor=[[[0]],
            [[1]],
            [[1]],
            [[0]]]

net=Network([2,2,2,1],["reLu","reLu","reLu"])

net.fit(input_Xor,output_Xor,0.1,200)

print(net.pred([[0],[0]]))
print(net.pred([[0],[1]]))
print(net.pred([[1],[0]]))
print(net.pred([[1],[1]]))