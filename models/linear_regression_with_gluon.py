import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy
import nltk
import sys

sys.path.insert(0, '../')

import argparse

import matplotlib
import matplotlib.pyplot as plt

import data.train_data as train_data

mx.random.seed(1)

data_context = mx.cpu()
model_context = mx.cpu()


DATA_SIZE = 50000
FEATURE_COUNT = 1000
EPOCHS = 100
LEARNING_RATE = .0001
BATCH_SIZE = 25

def plot(loss_sequence):
	plt.figure(num=None,figsize=(8, 6))
	plt.plot(loss_sequence)

	# Adding some bells and whistles to the plot
	plt.grid(True, which="both")
	plt.xlabel('epoch',fontsize=14)
	plt.ylabel('average loss',fontsize=14)

def train_section_class_classififer(section):

	data_loader, test_loader, data_size, num_outputs = train_data.get_data_loader(section)

	net = gluon.nn.Dense(1)
	params = net.collect_params()
	params.initialize(mx.init.Normal(sigma=1))
	squared_loss = gluon.loss.L2Loss()
	trainer = gluon.Trainer(params, 'sgd', {'learning_rate': LEARNING_RATE})

	num_batches = data_size / BATCH_SIZE
	loss_sequence = []

	for epoch in range(EPOCHS):
		cumulative_loss = 0

		for index, (data, label) in enumerate(data_loader):
			data = data.as_in_context(model_context)
			label = label.as_in_context(model_context).reshape((-1, 1))

			with autograd.record():
				output = net(data)
				loss = squared_loss(output, label)
			loss.backward()
			trainer.step(BATCH_SIZE)

			cumulative_loss += nd.mean(loss).asscalar()
		print("Cumulative loss: %s"%(cumulative_loss / data_size))
		loss_sequence.append(cumulative_loss)

	plot(loss_sequence)
	



parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train_section_class_classififer(section)

