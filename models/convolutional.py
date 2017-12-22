import mxnet as mx
from mxnet import nd, autograd, gluon
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../')

import data.train_data as train_data

mx.random.seed(1)

data_context = mx.cpu()
model_context = mx.cpu()


EPOCHS = 1000
LEARNING_RATE = .01
SMOOTHING_CONSTANT = .01

def relu(z):
	return nd.log(1 + nd.exp(z))

def softmax(y_linear):

	exp = nd.exp(y_linear - nd.max(y_linear))
	norms = exp.sum()
	softmax_output = exp/(norms + .0001)

	return softmax_output

def cross_entropy(prediction, label):
	return -(np.sum(prediction * np.log(label)))

def get_trainer(params):
	trainer = gluon.Trainer(params, 'sgd', {'learning_rate': LEARNING_RATE})

	return trainer

def initialize_params(net):
	params = net.collect_params()
	params.initialize(mx.init.Xavier(magnitude=2.24), ctx = model_context)

	return params

def initialize_net(num_outputs):
	full_connect_outputs = num_outputs
	net = gluon.nn.Sequential()

	with net.name_scope():
		net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, strides=1, activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=4, strides=1))

		net.add(gluon.nn.Conv2D(channels=50, kernel_size=4, strides=1,activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=4, strides=1))
		
		net.add(gluon.nn.Conv2D(channels=10, kernel_size=1, strides=1,activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=4, strides=1))

		net.add(gluon.nn.Conv2D(channels=5, kernel_size=1, strides=1,activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=4, strides=1))
		
		net.add(gluon.nn.Flatten())
		net.add(gluon.nn.Dense(full_connect_outputs, activation='relu'))
		net.add(gluon.nn.Dense(num_outputs))

	return net

def evaluation_accuracy(data_iterator, net):
	acc = mx.metric.Accuracy()
	for index, (data, label) in enumerate(data_iterator):
		data = data.as_in_context(model_context).reshape((train_data.BATCH_SIZE, 1, 25, 25))
		label = label.as_in_context(model_context)

		output = net(data)

		for output_index in range(len(output)):
			output[output_index] = softmax(output[output_index])

		acc.update(preds=output, labels=label)
	return acc.get()[1]

def train(section):
	print("training section %s"% section)

	print("getting training data")
	train, test, data_size, num_outputs = train_data.get_data_loader(section)

	print("initializing network with %s outputs"% num_outputs)
	net = initialize_net(num_outputs)

	print("collecting parameters")
	params = initialize_params(net)

	print("initializing loss function")
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, batch_axis=1)

	print("getting tainer")
	trainer = get_trainer(params)

	loss_sequence = []
	for epoch in range(EPOCHS):
		cumulative_loss = 0

		print("starting data enumeration")
		for index, (data, label) in enumerate(train):
			data = data.as_in_context(model_context).reshape((train_data.BATCH_SIZE, 1, 25, 25))
			print("data shape: %s, %s, %s, %s"% data.shape)

			label = label.as_in_context(model_context)

			with autograd.record():
				output = net(data)
				loss = softmax_cross_entropy(output, label)
				print("loss: %s"%loss)

			loss.backward()
			trainer.step(train_data.BATCH_SIZE)
			current_loss = nd.mean(loss).asscalar()
			moving_loss = (
								current_loss if((index == 0) and (epoch == 0)) 
							else 
								(1 - SMOOTHING_CONSTANT) * moving_loss + SMOOTHING_CONSTANT * current_loss)
			loss_sequence.append(moving_loss)
			
		test_accuracy = evaluation_accuracy(test, net)
		train_accuracy = evaluation_accuracy(train, net)

		print("Epoch %s loss: %s; Train Accc: %s; Test Acc: %s"% (epoch, moving_loss / data_size, train_accuracy, test_accuracy))
		

	print("saving model")
	file_name = "section_" + section + "_convolutional_model.params"
	net.save_params(file_name)

	plt.plot(loss_sequence)
	plt.grid(True, which="both")
	plt.xlabel('epoch',fontsize=14)
	plt.ylabel('average loss',fontsize=14)
	plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train(section)