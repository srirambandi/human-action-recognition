import pickle
import numpy as np
from keras.models import load_model
from i3d_inception import Inception_Inflated3d


classes = os.listdir('../kinetics_rgb/')
print(classes)
classes_dict = {i:classes[i] for i in range(len(classes))}

autism_classes = pickle.load("data/classes.p")
print(autism_classes)
autism_dict = {i:autism_classes[i] for i in range(len(autism_classes))}

rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_kinetics_and_autism', input_shape=(None, 224, 224, 3), classes=8)
rgb_model.load_weights("weights/rgb.h5")
#flow_model = Inception_Inflated3d(include_top=False, weights='flow_kinetics_and_autism', input_shape=(None, 224, 224, 2), classes=8)
#flow_model.load_weights("weights/flow.h5")

rgb_map = {}
rgb_classification = {}

for i in range(len(classes)):
        f = os.listdir("../kinetics_rgb/"+classes[i])
        preds = [0 for cc in range(8)]
        for file in f:
            x = np.load("../kinetics_rgb/"+classes[i]+"/"+file)
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
            rgb_logits = rgb_model.predict(x)
            sample_predictions = np.exp(rgb_logits) / np.sum(np.exp(rgb_logits))
            sorted_indices = np.argsort(sample_predictions)[::-1]
            pred_class = sorted_indices[0]
            preds[pred_class] = preds[pred_class] + 1
        rgb_classification[classes_dict[i]] = preds
        preds = [x/len(f) for x in preds]
        map_n = np.argmax(np.asarray(preds))
        if preds[map_n] >= 0.5:
            rgb_map[classes_dict[i]] = autism_dict[map_n]
