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


EPOCHS = 1
LEARNING_RATE = .01


def logistic(z):
	return 1. / (1. + nd.exp(-z))

def relu(z):
	return nd.log(1 + nd.exp(z))

def log_loss(output, y):
	yhat = relu(output)
	return -nd.sum(y * nd.log(yhat) + (1-y) * nd.log(1-yhat))

def initialize_params(net):
	params = net.collect_params()
	params.initialize(mx.init.Normal(sigma=1), ctx = model_context)

	return params

def softmax(y_linear):

	exp = nd.exp(y_linear - nd.max(y_linear))
	norms = exp.sum()
	softmax_output = exp/norms

	return softmax_output


def get_trainer(params):
	trainer = gluon.Trainer(params, 'sgd', {'learning_rate': LEARNING_RATE})

	return trainer

def evaluation_accuracy(data_iterator, net):
	acc = mx.metric.Accuracy()
	for index, (data, label) in enumerate(data_iterator):
		data = data.as_in_context(model_context)
		label = label.as_in_context(model_context)

		output = net(data)

		for output_index in range(len(output)):
			output[output_index] = softmax(output[output_index])
		print("prediction: %s; label: %s"% (output[0], label[0]))
		acc.update(preds=output, labels=label)
	return acc.get()[1]

def cross_entropy(prediction, label):
	return -(nd.sum(prediction * nd.log(label)))


def train(section):
	print("getting training data")
	train, test, data_size, num_outputs = train_data.get_data_loader(section)

	print("initializing network with %s outputs"% num_outputs)
	net = gluon.nn.Dense(num_outputs, activation='relu')

	print("collecting parameters")
	params = initialize_params(net)

	print("initializing loss function")
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=True, batch_axis=1)
	#softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

	print("getting tainer")
	trainer = get_trainer(params)

	loss_sequence = []

	print("starting training")
	for epoch in range(EPOCHS):
		cumulative_loss = 0

		print("starting data enumeration")
		for index, (data, label) in enumerate(train):
			data = data.as_in_context(model_context)
			#print("data: %s"% data)
			label = label.as_in_context(model_context)

			with autograd.record():
				output = net(data)
				loss = softmax_cross_entropy(output, label)
				print("loss: %s"%loss)
			loss.backward()
			trainer.step(train_data.BATCH_SIZE)
			cumulative_loss += nd.sum(loss).asscalar()
			loss_sequence.append(cumulative_loss)

		test_accuracy = evaluation_accuracy(test, net)
		train_accuracy = evaluation_accuracy(train, net)

		print("Epoch %s loss: %s; Train Accc: %s; Test Acc: %s"% (epoch, cumulative_loss / data_size, train_accuracy, test_accuracy))
		loss_sequence.append((epoch, cumulative_loss))
		plt.figure(num=None,figsize=(8, 6))
		plt.plot(epoch, cumulative_loss)

	print("saving model")
	file_name = "section_" + section + "_model.params"
	net.save_params(file_name)

parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train(section)
