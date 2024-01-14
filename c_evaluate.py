import pickle
import numpy as np
from keras.models import load_model
from model1 import Inception_Inflated3d

flow_test_path = '/home/mech/btech/me1130654/keras-kinetics-i3d/data/f_test.p'
rgb_test_path = '/home/mech/btech/me1130654/keras-kinetics-i3d/data/r_test.p'
label_test_path = '/home/mech/btech/me1130654/keras-kinetics-i3d/data/l_test.p'


def generate_arrays_from_file(data_path, labels_path):
        while True:
                data = pickle.load(open(data_path, "rb"))
                labels = pickle.load(open(labels_path, "rb"))
                for i in range(len(labels)):
                        x, y = data[i], labels[i]
                        yield x, y


rgb_data = pickle.load(open(rgb_test_path, "rb"))
flow_data = pickle.load(open(flow_test_path, "rb"))
labels = pickle.load(open(label_test_path, "rb"))


rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), classes=13)
rgb_model.load_weights('/home/mech/btech/me1130654/keras-kinetics-i3d/data/rgb.h5')
flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 7, 7, 832), classes=13)
flow_model.load_weights('/home/mech/btech/me1130654/keras-kinetics-i3d/data/flow.h5')

count = 0
for i in range(len(labels)):
	rgb_logits = rgb_model.predict(rgb_data[i])
	flow_logits = flow_model.predict(flow_data[i])
	sample_logits = rgb_logits + flow_logits
	sample_logits = sample_logits[0]
	sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
	sorted_indices = np.argsort(sample_predictions)[::-1]
	pred_class = sorted_indices[0]
	org_class = np.argmax(labels[i])
	if pred_class == org_class:
		count = count + 1

acc = count/len(labels)
acc_list = pickle.load(open('/home/mech/btech/me1130654/keras-kinetics-i3d/data/acc_list.p', "rb"))
acc_list.append(acc)
pickle.dump(acc_list, open('/home/mech/btech/me1130654/keras-kinetics-i3d/data/acc_list.p', "wb"))
print("accuracy now", acc)


# rgb_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/rgb1.h5')
# flow_model = load_model('/home/mech/btech/me1130654/keras-kinetics-i3d/models/flow1.h5')


# rgb_scores = rgb_model.evaluate_generator(generate_arrays_from_file(rgb_test_path, label_test_path), steps=200, verbose=0)
# print(rgb_scores)

# flow_scores = flow_model.evaluate_generator(generate_arrays_from_file(flow_test_path, label_test_path), steps=200, verbose=0)
# print(flow_scores)


