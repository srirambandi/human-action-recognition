import pickle
import numpy as np
from model1 import Inception_Inflated3d
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


flow_train_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/new_total_f_train"+str(i)+".p" for i in range(6)]
rgb_train_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/new_total_r_train"+str(i)+".p" for i in range(6)]
label_train_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/new_total_l_train"+str(i)+".p" for i in range(6)]

def generate_arrays_from_file(data, labels):
	while True:
		for i in range(len(labels)):
			x, y = data[i], labels[i]
			yield x, y


earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')

for i in range(6):

        rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), endpoint_logit=False, classes=14)
        sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
        rgb_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        rgb_model.summary()

        rgb_train_data = pickle.load(open(rgb_train_path[i], "rb"))
        label_train_data = pickle.load(open(label_train_path[i], "rb"))
        steps = len(label_train_data)
        rgb_model.fit_generator(generate_arrays_from_file(rgb_train_data, label_train_data), steps_per_epoch=steps, epochs=300)


        rgb_model.save("/home/mech/btech/me1130654/keras-kinetics-i3d/data/cross_val6_1/new_total_rgb_cv"+str(i)+".h5")

        print("RGB Model", i, "saved \n")
