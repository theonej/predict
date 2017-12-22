import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy
import nltk
import sys

sys.path.insert(0, '../')

import argparse
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

import data.train_data as train_data

mx.random.seed(1)

data_context = mx.cpu()
model_context = mx.cpu()


EPOCHS = 1000
LEARNING_RATE = .0001


def net(X, W, b):
	return mx.nd.dot(X, W) + b

def squared_loss(yhat, y):
	return nd.mean((yhat - y) ** 2)

#Schocastic Gradient Descent
def SGD(params, lr):
	for param in params:
		param[:] = param - lr * param.grad

def plot(loss_sequence):
	plt.figure(num=None,figsize=(8, 6))
	plt.plot(loss_sequence)

	# Adding some bells and whistles to the plot
	plt.grid(True, which="both")
	plt.xlabel('epoch',fontsize=14)
	plt.ylabel('average loss',fontsize=14)

def train_section_class_classififer(section):

	data_loader, test_loader, data_size, num_outputs = train_data.get_data_loader(section)

	W = nd.random_normal(shape=(FEATURE_COUNT, num_outputs), ctx=model_context)
	b = nd.random_normal(shape=num_outputs, ctx=model_context)
	
	params=[W, b]

	for param in params:
		param.attach_grad()

	num_batches = data_size / train_data.BATCH_SIZE
	loss_sequence = []

	for epoch in range(EPOCHS):
		cumulative_loss = 0

		for index, (data, label) in enumerate(data_loader):
			data = data.as_in_context(model_context)
			label = label.as_in_context(model_context).reshape((-1, 1))

			with autograd.record():
				output = net(data, W, b)
				loss = squared_loss(output, label)
			loss.backward()
			SGD(params, LEARNING_RATE)
			cumulative_loss += loss.asscalar()
		print("Cumulative loss: %s"%(cumulative_loss / num_batches))
		loss_sequence.append(cumulative_loss)

	plot(loss_sequence)
	



parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train_section_class_classififer(section)

