import os
import sys
import pickle
import numpy as np
from random import shuffle

classes = os.listdir('/home/prathosh/ram/compressed_binary/Misc/rgb/')
print(classes)
classfiles = [os.listdir('/home/prathosh/ram/compressed_binary/Misc/flow_threads/'+classes[i]) for i in range(len(classes))]
print([len(i) for i in classfiles])
classes_dict = {i:classes[i] for i in range(len(classes))}
print(classes_dict)


# def chunk_6(data):
#         s = (len(data)/6)
#         l = [data[int(i*s):int(i*s+s)] for i in range(6)]
#         return l

for i in range(len(classes)):
	shuffle(classfiles[i])
	# classfiles[i] = chunk_1(classfiles[i])


print([len(classfiles[i]) for i in range(len(classfiles))])

# sys.exit()

for t in range(1):
        flow_train = []
        rgb_train = []
        labels_train = []
        flow_test = []
        rgb_test = []
        labels_test = []
        for i in range(len(classes)):
                length = len(classfiles[i])
                train_classfiles = classfiles[i][:int(0.8*length)]
                test_classfiles = classfiles[i][int(0.8*length):]
                for file in train_classfiles:
                        x = np.load('/home/prathosh/ram/compressed_binary/Misc/flow_threads/'+classes[i]+'/'+file)
                        flow_train.append(x)
                        x = np.load('/home/prathosh/ram/compressed_binary/Misc/rgb/'+classes[i]+'/'+file)
                        rgb_train.append(x)
                        y = np.zeros((1, 2))
                        y[0, i] = 1
                        labels_train.append(y)
                for file in test_classfiles:
                        x = np.load('/home/prathosh/ram/compressed_binary/Misc/flow_threads/'+classes[i]+'/'+file)
                        flow_test.append(x)
                        x = np.load('/home/prathosh/ram/compressed_binary/Misc/rgb/'+classes[i]+'/'+file)
                        rgb_test.append(x)
                        y = np.zeros((1, 2))
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


        pickle.dump(labels_shuf_train, open("/home/prathosh/ram/data/binary/Misc/l_train"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_train, open("/home/prathosh/ram/data/binary/Misc/f_train"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_train, open("/home/prathosh/ram/data/binary/Misc/r_train"+str(t)+".p", "wb"))
        pickle.dump(labels_shuf_test , open("/home/prathosh/ram/data/binary/Misc/l_test"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_test, open("/home/prathosh/ram/data/binary/Misc/f_test"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_test, open("/home/prathosh/ram/data/binary/Misc/r_test"+str(t)+".p", "wb"))
        print("data created :", t)
