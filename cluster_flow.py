import pickle
import numpy as np
from keras.models import load_model
from i3d_inception import Inception_Inflated3d

#flow_test_path = ["../crossval5_8/f_test"+str(i)+".p" for i in range(5)]
#rgb_test_path = ["../crossval5_8/r_test"+str(i)+".p" for i in range(5)]
#label_test_path = ["../crossval5_8/l_test"+str(i)+".p" for i in range(5)]
# LABEL_MAP_PATH = '/home/mech/btech/me1130654/keras-kinetics-i3d/data/autism_label_map.txt'

classes = os.listdir('../kinetics_rgb/')
print(classes)

# autism_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

def generate_arrays_from_file(data_path, labels_path):
        while True:
                data = pickle.load(open(data_path, "rb"))[0]
                labels = pickle.load(open(labels_path, "rb"))[0]
                for i in range(len(labels)):
                        #x, y = data[i], labels[i]
                        x, y = np.load(data[i]), labels[i]
                        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                        yield x, y

rgb_data = []
flow_data = []
labels = []
#for t in range(5):
#        rgb_data.append(pickle.load(open(rgb_test_path[t], "rb")))
#        flow_data.append(pickle.load(open(flow_test_path[t], "rb")))
#        labels.append(pickle.load(open(label_test_path[t], "rb")))

acc_list = []

#rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_kinetics_and_autism', input_shape=(None, 224, 224, 3), classes=8)
rgb_model = load_model("weights/rgb.h5")
#flow_model = Inception_Inflated3d(include_top=False, weights='flow_kinetics_and_autism', input_shape=(None, 224, 224, 2), classes=8)
flow_model = load_model("weights/flow.h5")
count = 0
y_pred = []
y_true = []
for i in range(len(labels[t])):
        x = np.load(rgb_data[t][i])
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        rgb_logits = rgb_model.predict(x)
        x = np.load(flow_data[t][i])
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        flow_logits = flow_model.predict(x)
        sample_logits = rgb_logits + flow_logits
        sample_logits = sample_logits[0]
        sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
        sorted_indices = np.argsort(sample_predictions)[::-1]
        pred_class = sorted_indices[0]
        true_class = np.argmax(labels[t][i])

        y_pred.append(pred_class)
        y_true.append(true_class)
        if pred_class == true_class:
                count = count + 1

pickle.dump(y_pred, open("y_pred"+str(t)+".p", "wb"))
pickle.dump(y_true, open("y_true"+str(t)+".p", "wb"))
acc = count/len(labels[t])
acc_list.append(acc)
# print("fold: {}, no. of samples: {}, acc: {}".format(t, len(labels[t]), acc))

#print("All folds individual accuracies:", acc_list)
print("Average accuracy:", sum(acc_list)/len(acc_list))


# rgb_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/rgb1.h5')
# flow_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/flow1.h5')


# rgb_scores = rgb_model.evaluate_generator(generate_arrays_from_file(rgb_test_path, label_test_path), steps=200, verbose=0)
# print(rgb_scores)

# flow_scores = flow_model.evaluate_generator(generate_arrays_from_file(flow_test_path, label_test_path), steps=200, verbose=0)
# print(flow_scores)
