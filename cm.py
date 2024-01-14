import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred = []
y_true = []
classes = pickle.load(open("classes.p", "rb"))

for i in range(1):
	y_pred.append(pickle.load(open("scores/exp2_y_pred"+str(i)+".p", "rb")))
	y_true.append(pickle.load(open("scores/exp2_y_true"+str(i)+".p", "rb")))

for fold in range(1):
	cm = confusion_matrix(y_true[fold], y_pred[fold])
	np.set_printoptions(precision=2)

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix - Fold: "+str(fold))
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	normalize = False
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.show()
