import pickle
import numpy as np
from model import Inception_Inflated3d
from keras.optimizers import SGD

flow_train_path = '/home/mech/btech/me1130654/scratch/flow_train.p'
rgb_train_path = '/home/mech/btech/me1130654/scratch/rgb_train.p'
label_train_path = '/home/mech/btech/me1130654/scratch/labels_train.p'

def generate_arrays_from_file(data_path, labels_path):
	while True:
		data = pickle.load(open(data_path, "rb"))
		labels = pickle.load(open(labels_path, "rb"))
		for i in range(len(labels)):
			x, y = data[i], labels[i]
			yield x, y


rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), endpoint_logit=False, classes=13)
sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
rgb_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
rgb_model.summary()


# generator = generate_arrays_from_file(rgb_train_path, label_train_path)
rgb_model.fit_generator(generate_arrays_from_file(rgb_train_path, label_train_path), steps_per_epoch=1009, epochs=1000)


rgb_model.save('models/rgb1.h5')
