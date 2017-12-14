import os
import numpy as np
import cv2
from utils import parse_image, parse_canny_image
import csv
import tensorflow as tf

import random
from cnn_model import cnn
from canny_model import canny

classes = {'up': 0, 'right': 1, 'left': 2}
num_classes = 3
train_split = 0.8 
cwd = os.getcwd()
data_path = '/home/aasmund/Documents/universe/recordings/parsed/'



def get_data(model):

	data = []
	input_data = []
	output_data = []
	direction_folders = os.listdir(data_path)
	for direction in direction_folders:
		l = len(data)
		images_path = os.path.join(data_path, direction)
		for image in os.listdir(images_path):
			# 0 entails grayscale image
			image_path = os.path.join(images_path, image)
			# print(image_path)
			img = cv2.imread(image_path, 0)

			# input_data.append(parse_image(img))
			# output_data.append(classes[str(direction).lower()])
			if(model=='canny'):
				data.append((parse_canny_image(img), classes[str(direction).lower()]))
			else:
				data.append((parse_image(img), classes[str(direction).lower()]))

		print(len(data)-l)

	
	random.shuffle(data)

	split = int(len(data)*train_split)
	train = data[:split]
	test = data[split:]

	return (train, test)
	# return np.array(input_data, dtype=np.float32), np.array(output_data, dtype=np.int32)

def classify(img, model_type):
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": img},
			num_epochs=1,
			shuffle=False)

	if(model_type=='canny'):
		fn_classifier = tf.estimator.Estimator(
			model_fn = canny,
			model_dir = cwd+'/ikt440-project/tmp/canny_model')		
	else:
		fn_classifier = tf.estimator.Estimator(
			model_fn = cnn,
			model_dir = cwd+'/ikt440-project/tmp/cnn_model')


	predictions = fn_classifier.predict(input_fn = predict_input_fn)
	print('predictions are: ')
	ps = []
	for i,k in enumerate(predictions):
		ps.append(k)
	print(ps)
	return ps



def main(model):	
	tf.logging.set_verbosity(tf.logging.INFO)
	data = get_data(model)
	train = data[0]
	test = data[1]
	print(type(train))
	print('length of training data: ', len(train))

	train_x = np.array([train[i][0] for i in range(len(train))], dtype=np.float32)
	train_y = np.array([train[i][1] for i in range(len(train))], dtype=np.int32)

	print(train_y)


	## Instantiate the model
	if model == 'canny':
		fn_classifier = tf.estimator.Estimator(
			model_fn = canny,
			model_dir = cwd+'/tmp/canny_model')
	else:
		fn_classifier = tf.estimator.Estimator(
			model_fn = cnn,
			model_dir = cwd+'/tmp/cnn_model')

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	summ = tf.summary.merge_all_summaries()

	## Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {'x':train_x},
		y = train_y,
		batch_size = 10,
		num_epochs = None,
		shuffle = True)


	fn_classifier.train(
		input_fn = train_input_fn,
		steps = 5000,
		hooks=[logging_hook])

	writer = tf.summary.FileWriter('/tmp/tensorboard/')
	## Test the model

	test_x = np.array([test[i][0] for i in range(len(test))], dtype=np.float32)
	test_y = np.array([test[i][1] for i in range(len(test))], dtype=np.int32)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {'x': test_x},
		y = test_y,
		num_epochs = 1,
		shuffle = False)	

	
	res = fn_classifier.evaluate(input_fn = test_input_fn)

	print('classifcation on test is: ')
	print(res)

if __name__=="__main__":
	print('input model type, either cnn or canny')
	m = input()
	if(m=='cnn' or m=='canny'):
		main(m)
	else:
		print('try again with input cnn or canny')
		exit()

