import sys
import subprocess
import pickle 
import numpy as np
from random import shuffle
from model import Inception_Inflated3d
from keras.optimizers import SGD
from keras.models import load_model


# train data
flow_train_path = '/home/mech/btech/me1130654/scratch/flow_train.p'
rgb_train_path = '/home/mech/btech/me1130654/scratch/rgb_train.p'
label_train_path = '/home/mech/btech/me1130654/scratch/labels_train.p'


# test data
flow_test_path = '/home/mech/btech/me1130654/scratch/flow_test.p'
rgb_test_path = '/home/mech/btech/me1130654/scratch/rgb_test.p'
label_test_path = '/home/mech/btech/me1130654/scratch/labels_test.p'


flow_data = pickle.load(open(flow_train_path, "rb"))[0] + pickle.load(open(flow_test_path, "rb"))[0]
rgb_data = pickle.load(open(rgb_train_path, "rb"))[0] + pickle.load(open(rgb_test_path, "rb"))[0]
label_data = pickle.load(open(label_train_path, "rb"))[0] + pickle.load(open(label_test_path, "rb"))[0]



def chunk_5(data):
	s = (len(data)/5) + 1
	l = [data[int(i*s):int(i*s+s)] for i in range(5)]
	return l


f = chunk_5(flow_data)
r = chunk_5(rgb_data)
l = chunk_5(label_data)

shuffle(f)
shuffle(r)
shuffle(l)

for i in range(5):
	f_test = f[i]
	f_train = []
	for j in range(5):
		if j != i:
			f_train = f_train + f[j]
	r_test = r[i]
	r_train	= []
	for j in range(5):
		if j !=	i:
			r_train	= r_train + r[j]
	l_test = l[i]
	l_train	= []
	for j in range(5):
		if j !=	i:
			l_train	= l_train + l[j]
	pickle.dump(f_train, open("/home/mech/btech/me1130654/scratch/cross_val/f_train"+str(i)+".p", "wb"))
	print("f train")
	pickle.dump(f_test, open("/home/mech/btech/me1130654/scratch/cross_val/f_test"+str(i)+".p", "wb"))
	print("f test")
	pickle.dump(r_train, open("/home/mech/btech/me1130654/scratch/cross_val/r_train"+str(i)+".p", "wb"))
	print("r train")
	pickle.dump(r_test, open("/home/mech/btech/me1130654/scratch/cross_val/r_test"+str(i)+".p", "wb"))
	print("r test")
	pickle.dump(l_train, open("/home/mech/btech/me1130654/scratch/cross_val/l_train"+str(i)+".p", "wb"))
	print("l train")
	pickle.dump(l_test, open("/home/mech/btech/me1130654/scratch/cross_val/l_test"+str(i)+".p", "wb"))
	print("l test")
	print("pass :", i, "---- data created...")
