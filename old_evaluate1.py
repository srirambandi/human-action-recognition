import pickle
import numpy as np
from keras.models import load_model
from model1 import Inception_Inflated3d

flow_test_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/f_test"+str(i)+".p" for i in range(6)]
rgb_test_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/r_test"+str(i)+".p" for i in range(6)]
label_test_path = ["/home/mech/btech/me1130654/scratch/cross_val6_1/l_test"+str(i)+".p" for i in range(6)]
# LABEL_MAP_PATH = '/home/mech/btech/me1130654/keras-kinetics-i3d/data/autism_label_map.txt'

# autism_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

def generate_arrays_from_file(data_path, labels_path):
        while True:
                data = pickle.load(open(data_path, "rb"))[0]
                labels = pickle.load(open(labels_path, "rb"))[0]
                for i in range(len(labels)):
                        x, y = data[i], labels[i]
                        yield x, y

rgb_data = []
flow_data = []
labels = []
for t in range(6):
        rgb_data.append(pickle.load(open(rgb_test_path[t], "rb")))
        flow_data.append(pickle.load(open(flow_test_path[t], "rb")))
        labels.append(pickle.load(open(label_test_path[t], "rb")))

acc_list = []

for t in range(6):
        rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), classes=13)
        rgb_model.load_weights("/home/mech/btech/me1130654/keras-kinetics-i3d/data/cross_val6_1/rgb_cv"+str(t)+".h5")
        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), classes=13)
        flow_model.load_weights("/home/mech/btech/me1130654/keras-kinetics-i3d/data/cross_val6_1/flow_cv"+str(t)+".h5")
        count = 0
        for i in range(len(labels[t])):
                rgb_logits = rgb_model.predict(rgb_data[t][i])
                flow_logits = flow_model.predict(flow_data[t][i])
                sample_logits = rgb_logits + flow_logits
                sample_logits = sample_logits[0]
                sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
                sorted_indices = np.argsort(sample_predictions)[::-1]
                pred_class = sorted_indices[0]
                org_class = np.argmax(labels[t][i])

                if pred_class == org_class:
                        count = count + 1

        acc = count/len(labels[t])
        acc_list.append(acc)

print("All folds individual accuracies:", acc_list)
print("Average accuracy:", sum(acc_list)/len(acc_list))


# rgb_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/rgb1.h5')
# flow_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/flow1.h5')


# rgb_scores = rgb_model.evaluate_generator(generate_arrays_from_file(rgb_test_path, label_test_path), steps=200, verbose=0)
# print(rgb_scores)

# flow_scores = flow_model.evaluate_generator(generate_arrays_from_file(flow_test_path, label_test_path), steps=200, verbose=0)
# print(flow_scores)


