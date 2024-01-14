import pickle
import numpy as np
from model import Inception_Inflated3d
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

flow_train_path = '/home/mech/btech/me1130654/scratch/flow_train.p'
rgb_train_path = '/home/mech/btech/me1130654/scratch/rgb_train.p'
label_train_path = '/home/mech/btech/me1130654/scratch/labels_train.p'

def generate_arrays_from_file(data_path, labels_path):
	while True:
		data = pickle.load(open(data_path, "rb"))[0]
		labels = pickle.load(open(labels_path, "rb"))[0]
		for i in range(len(labels)):
			x, y = data[i], labels[i]
			yield x, y


earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')


flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), endpoint_logit=False, classes=13)
sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
flow_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
flow_model.summary()


flow_model.fit_generator(generate_arrays_from_file(flow_train_path, label_train_path), steps_per_epoch=1009, epochs=600)


flow_model.save('/home/mech/btech/me1130654/keras-kinetics-i3d/models/flow1.h5')

print("Model saved")
