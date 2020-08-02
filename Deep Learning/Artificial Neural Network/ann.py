import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# no. of hidden neurons (nh) = (2/3)* (no.of input neurons + no.of output neurons)
nh = ((2/3)*(11+1))
classifier.add(Dense(activation='relu', units=nh, kernel_initializer='uniform', input_dim=11))

# add 2nd hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

classifier.compile()
classifier.fit()