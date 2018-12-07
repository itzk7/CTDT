import keras
from keras.models import Sequential
from keras.layers import Dense


num_classes = 2

def getVGG16(flag = True):
	if flag == True:
		vgg16 = keras.applications.VGG16(weights = None)
	else:
		vgg16 = keras.applications.VGG16(weights = None, include_top = False)

	model = Sequential()
	for layer in vgg16.layers:
		model.add(layer)
	model.layers.pop()
	model.add(Dense(2, activation='softmax', name='predictions'))

	return model
model = getVGG16(False)

