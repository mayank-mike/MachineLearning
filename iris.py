import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense
import random

dataset = numpy.loadtxt("new.csv", delimiter=",")

random.shuffle(dataset)

X = dataset[:,0:-1]
Y = dataset[:,-1]

test_X = X[120:,:]
test_Y = Y[120:]

X = X[:120,:]
Y = Y[:120]

# create model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=20)

# evaluate the model
scores = model.evaluate(test_X, test_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





