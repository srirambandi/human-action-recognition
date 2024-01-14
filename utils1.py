import os
import sys
import pickle
import numpy as np
from random import shuffle

classes = os.listdir('/home/mech/btech/me1130654/scratch/total_compressed1_13/rgb/')
print(classes)
classfiles = [os.listdir('/home/mech/btech/me1130654/scratch/total_compressed1_13/flow_threads/'+classes[i]) for i in range(len(classes))]
print([len(i) for i in classfiles])
classes_dict = {i:classes[i] for i in range(len(classes))}
print(classes_dict)


def chunk_6(data):
        s = (len(data)/6)
        l = [data[int(i*s):int(i*s+s)] for i in range(6)]
        return l

for i in range(len(classes)):
	shuffle(classfiles[i])
	classfiles[i] = chunk_6(classfiles[i])


print([len(classfiles[i]) for i in range(len(classfiles))])

# sys.exit()

for t in range(6):
        flow_train = []
        rgb_train = []
        labels_train = []
        flow_test = []
        rgb_test = []
        labels_test = []
        for i in range(len(classes)):
                test_classfiles = classfiles[i][t]
                train_classfiles = []
                for tt in range(6):
                        if tt != t:
                                train_classfiles = train_classfiles + classfiles[i][tt]
                for file in train_classfiles:
                        x = np.load('/home/mech/btech/me1130654/scratch/total_compressed1_13/flow_threads/'+classes[i]+'/'+file)
                        flow_train.append(x)
                        x = np.load('/home/mech/btech/me1130654/scratch/total_compressed1_13/rgb/'+classes[i]+'/'+file)
                        rgb_train.append(x)
                        y = np.zeros((1, 13))
                        y[0, i] = 1
                        labels_train.append(y)
                for file in test_classfiles:
                        x = np.load('/home/mech/btech/me1130654/scratch/total_compressed1_13/flow_threads/'+classes[i]+'/'+file)
                        flow_test.append(x)
                        x = np.load('/home/mech/btech/me1130654/scratch/total_compressed1_13/rgb/'+classes[i]+'/'+file)
                        rgb_test.append(x)
                        y = np.zeros((1, 13))
                        y[0, i] = 1
                        labels_test.append(y)
        print("data split done -", t)
	
        flow_shuf_train = []
        rgb_shuf_train = []
        labels_shuf_train = []
        flow_shuf_test = []
        rgb_shuf_test = []
        labels_shuf_test = []
        index_shuf = [i for i in range(len(labels_train))]
        shuffle(index_shuf)
        flow_shuf_train = [flow_train[i] for i in index_shuf]
        rgb_shuf_train = [rgb_train[i] for i in index_shuf]
        labels_shuf_train = [labels_train[i] for i in index_shuf]
        index_shuf = [i for i in range(len(labels_test))]
        shuffle(index_shuf)
        flow_shuf_test = [flow_test[i] for i in index_shuf]
        rgb_shuf_test = [rgb_test[i] for i in index_shuf]
        labels_shuf_test = [labels_test[i] for i in index_shuf]


        pickle.dump(labels_shuf_train, open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_l_train"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_train, open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_f_train"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_train, open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_r_train"+str(t)+".p", "wb"))
        pickle.dump(labels_shuf_test , open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_l_test"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_test, open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_f_test"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_test, open("/home/mech/btech/me1130654/scratch/cross_val6_1/13_total_r_test"+str(t)+".p", "wb"))
        print("data created :", t)
