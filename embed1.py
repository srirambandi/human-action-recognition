import os
import numpy as np
from i3d_inception import Inception_Inflated3d
from keras import backend as K

classes = os.listdir('/home/mech/btech/me1130654/scratch/total_data/rgb/')
print(classes)
classfiles = [os.listdir('/home/mech/btech/me1130654/scratch/total_data/flow_threads/'+classes[i]) for i in range(len(classes))]
print([len(i) for i in classfiles])
classes_dict = {i:classes[i] for i in range(len(classes))}
print(classes_dict)

NUM_FRAMES = None
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2
NUM_CLASSES = 13

n = 153

rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS), classes=NUM_CLASSES)


flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS), classes=NUM_CLASSES)


for i in range(len(classes)):
        for file in classfiles[i]:
                x = np.load('/home/mech/btech/me1130654/scratch/total_data/flow_threads/'+classes[i]+'/'+file)
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                get_nth_layer_output = K.function([flow_model.layers[0].input], [flow_model.layers[n].output])
                layer_output = get_nth_layer_output([x])[0]
                np.save('/home/mech/btech/me1130654/scratch/total_compressed1_13/flow_threads/'+classes[i]+'/'+file, layer_output)
                x = np.load('/home/mech/btech/me1130654/scratch/total_data/rgb/'+classes[i]+'/'+file)
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                get_nth_layer_output = K.function([rgb_model.layers[0].input], [rgb_model.layers[n].output])
                layer_output = get_nth_layer_output([x])[0]
                np.save('/home/mech/btech/me1130654/scratch/total_compressed1_13/rgb/'+classes[i]+'/'+file, layer_output)
        print(classes[i])
