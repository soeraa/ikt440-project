import tensorflow as tf
import numpy as np


def cnn(features, labels, mode):
	# input_layer = tf.placeholder("float", [None, 120, 160, 1], name='input')
	input_layer = tf.reshape(features['x'], [-1, 120, 160, 1])
	conv1 = tf.layers.conv2d(
	  inputs=input_layer,
	  filters=32,
	  kernel_size=[8, 8],
	  padding="same",
	  activation=tf.nn.relu)

	pool1 = tf.layers.max_pooling2d(
		inputs=conv1, 
		pool_size=[2, 2], 
		strides=2)
	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=64,
	  kernel_size=[4, 4],
	  padding="same",
	  activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2, 
		pool_size=[2, 2], 
		strides=2)
	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 76800])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
	  inputs=dense, rate=0.4)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=3)
	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  # `logging_hook`.
	  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# print('input layer ', input_layer.shape)
	# print('conv 1 ',conv1.shape)
	# print('pool 1 ', pool1.shape)
	# print('conv 2 ',conv2.shape)
	# print('pool 2 ',pool2.shape)
	# print('flatten ',pool2_flat.shape)
	# print('dense ',dense.shape)
	# print('logits', logits.shape)


	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	## one-hot encoding
	y = tf.one_hot(labels, 3)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	train_step = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

	return tf.estimator.EstimatorSpec(mode=mode,loss=loss, train_op=train_step)


