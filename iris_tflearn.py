import tflearn
import random
import numpy
import matplotlib.pyplot as plt

dataset = numpy.loadtxt("new.csv", delimiter=",")

random.shuffle(dataset)

def change(x):
    temp = numpy.full((len(x), 3), 0)
    for i in range(len(x)):
        temp[i][int(x[i])] = 1.0
    return temp

def maximus(x):
    if x[0]>x[1] and x[0]>x[2]:
        return 0
    elif x[1]>x[0] and x[1]>x[2]:
        return 1
    else :
        return 2


X = dataset[:,0:-1]
Y = dataset[:,-1]

test_X = X[120:,:]
test_Y = Y[120:]
test_Y_change = change(test_Y)

X = X[:120,:]
Y = Y[:120]
Y_change = change(Y)

# Build neural network
net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 64,activation="relu")
net = tflearn.fully_connected(net, 64,activation="sigmoid")
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(X, Y_change, n_epoch=100, batch_size=16, show_metric=True)

result = model.predict(test_X)

#plotting graph
colouring = {0:'r', 1:'g', 2:'b'}
count=0
for i in range(30):
    plt.scatter(test_X[i][0],test_X[i][1],c=colouring[int(test_Y[i])],marker=">")
    plt.scatter(test_X[i][0],test_X[i][1],c=colouring[int(maximus(result[i]))],marker="<")
    if int(test_Y[i]) == int(maximus(result[i])):
        count+=1

Accuracy=(count/30)*100
plt.title("Accuracy: "+str(Accuracy))

plt.show()














