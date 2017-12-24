import mxnet as mx
from mxnet import nd, autograd, gluon
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import data.train_data as train_data

parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train_features, train_labels, test_features, test_labels, data_size, num_outputs = train_data.get_class_labeled_data(section, 30)

train_numpy = np.array(train_features)

label_values = []
for row in range(len(train_labels)):
	index = train_labels[row].index(1)
	if(index != 2):
		index = 0
	else:
		index = 1
	label_values.append(index)


label_numpy = np.array(label_values)
print(label_numpy)

np.savetxt('./octave/data/labels.mtx', label_numpy)
np.savetxt('./octave/data/train.mtx', train_numpy)