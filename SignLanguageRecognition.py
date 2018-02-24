from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras.layers import Dense, Activation, Input
from keras.models import Model
from sklearn.utils import shuffle
import numpy as np
import os
import time


# Initialize data
print('Initializing data')
path = os.getcwd()
data_path = path + '/signs_dataset'
data_dir_list = os.listdir(data_path)

data = []

for pic in data_dir_list:
	if(pic != '.DS_Store'):
		img_path = data_path + '/' + pic
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		data.append(x)

data = np.array(data)
data = np.rollaxis(data, 1, 0)
data = data[0]
# print(data.shape)


# Initialize labels
print('Initializing labels')
num_classes = 24
num_samples = 1680#data.shape[0]
labels = np.ones((num_samples,1), dtype='int64')

for i in range(num_classes):
	if(i == num_classes - 1):
		labels[70*i:] = i
	else: 
		labels[70*i:70*(i+1)] = i

alphabet = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s',
			't','u','v','w','x','y']


# Convert labels to one-hot
Y = np_utils.to_categorical(labels, num_classes)


# Shuffle data
X,Y = shuffle(data,Y, random_state=0)

# Split data
X_train = X[:1344]
X_test = X[1344:]
Y_train = Y[:1344]
Y_test = Y[1344:]


print('Defining model')
# input tensor
image_input = Input(shape=(224,224,3))

model = VGG16(include_top = True, weights = 'imagenet', input_tensor = image_input)
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
custom_model = Model(image_input, out)


for layer in custom_model.layers[:-1]:
	layer.trainable = False


custom_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

print('Running model')
t = time.time()
hist = custom_model.fit(X_train, Y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, Y_test))
print('Training time: %s' % (time.time() - t))
(loss, accuracy) = custom_model.evaluate(X_test, Y_test, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
























