import tensorflow as tf
import numpy as np


def canny(features, labels, mode):
	# input_layer = tf.placeholder("float", [None, 120, 160, 1], name='input')
	# input_layer = tf.reshape(features['x'], [-1, 64, 64, 1])

	flat = tf.reshape(features['x'], [-1, 4096])
	
	dense1 = tf.layers.dense(inputs=flat, units=4096, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(
	  	inputs=dense1, rate=0.4)
	dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(
	  	inputs=dense2, rate=0.4)
	dense3 = tf.layers.dense(inputs=dense2, units=1024, activation=tf.nn.relu)
	dropout3 = tf.layers.dropout(
		inputs = dense3, rate=0.3)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout3, units=3)
	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  # `logging_hook`.
	  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# print('flatten ',flat.shape)
	# print('dense ', dense1.shape)
	# print('dense ', dense2.shape)
	# print('dense ', dense3.shape)



	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	## one-hot encoding
	y = tf.one_hot(labels, 3)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train_step = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

	tf.summary.scalar("Canny Loss", loss)

	correct_prediction = tf.equal(tf.argmax(predictions['probabilities'], 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("Canny Accuracy", accuracy)

	summ = tf.summary.merge_all()

	return tf.estimator.EstimatorSpec(mode=mode,loss=loss, train_op=train_step, evalutaion_hooks=summary)


