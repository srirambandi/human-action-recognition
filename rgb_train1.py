import pickle
import numpy as np
from model1 import Inception_Inflated3d
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

flow_train_path = '/home/mech/btech/me1130654/scratch/flow_train1.p'
rgb_train_path = '/home/mech/btech/me1130654/scratch/rgb_train1.p'
label_train_path = '/home/mech/btech/me1130654/scratch/labels_train1.p'

def generate_arrays_from_file(data_path, labels_path):
	while True:
		data = pickle.load(open(data_path, "rb"))[0]
		labels = pickle.load(open(labels_path, "rb"))[0]
		for i in range(len(labels)):
			x, y = data[i], labels[i]
			yield x, y


earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')


rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), endpoint_logit=False, classes=13)
sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
rgb_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
rgb_model.summary()


rgb_model.fit_generator(generate_arrays_from_file(rgb_train_path, label_train_path), steps_per_epoch=1009, epochs=200)


rgb_model.save('/home/mech/btech/me1130654/keras-kinetics-i3d/models/rgb2.h5')

print("Model saved")
