import keras.optimizers as opt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

def CNNmodel(num_emotions) :
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(48,48,1), activation='relu', padding='valid'))
	model.add(Conv2D(64, (5,5), activation='relu', padding='valid'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(128, (5, 5), activation='relu', padding='valid'))
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_emotions, kernel_initializer='normal', activation='softmax'))
	return model
