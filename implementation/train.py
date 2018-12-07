from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import cv2

input_shape = (150,150,3)
nClasses = 1

train_path = '/home/kesavan/Desktop/CTDT/dataset/train/'
test_path = '/home/kesavan/Desktop/CTDT/dataset/test/'

def load_images(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images

def createModel():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
 
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
 
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nClasses, activation='softmax'))
	 
	return model


if __name__ == '__main__':
	
	model = createModel();
	model.compile(loss='mean_squared_error', optimizer='sgd');
	model.summary()

	train_datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
		fill_mode='nearest')

	test_datagen = ImageDataGenerator(rescale=1./255)
	
	train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
	
	# i = 0;
	# for batch in train_datagen.flow(train_generator, batch_size = 1, save_to_dir = 'augmented', save_prefix = 'IMG', save_format = 'jpg'):
	# 	if(i == 5):
	# 		break;
	# 	i = i + 1;
	model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800)

	model.save('model.h5')