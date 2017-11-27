import os
import numpy as np
import cv2
import utils
import csv
import tensorflow as tf
from model import cnn

classes = {'up': 0, 'right': 1, 'left': 2}
num_classes = 3
train_split = 0.8 
cwd = os.getcwd()
data_path = '/home/aasmund/Documents/universe/recordings/images'

def parse_image(img):
	img = cv2.resize(img, (120, 160))
	img = img.astype(np.float32)
	img *= (1.0 / 255.0)
	img = np.reshape(img, [120, 160, 1])
	return img



def get_data():

	input_data = []
	output_data = []
	direction_folders = os.listdir(data_path)
	for direction in direction_folders:
		l = len(input_data)
		images_path = os.path.join(data_path, direction)
		for image in os.listdir(images_path):
			# 0 entails grayscale image
			image_path = os.path.join(images_path, image)
			img = cv2.imread(image_path, 0)
			input_data.append(parse_image(img))
			output_data.append(classes[str(direction).lower()])
	return np.array(input_data, dtype=np.float32), np.array(output_data, dtype=np.int32)

def classify(img):
	cnn_classifier = tf.estimator.Estimator(
		model_fn = cnn,
		model_dir = cwd+'/ikt440-project/tmp/cnn_model')

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": img},
		num_epochs=1,
		shuffle=False)

	predictions = cnn_classifier.predict(input_fn = predict_input_fn)
	print('predictions are: ')
	ps = []
	for i,k in enumerate(predictions):
		ps.append(k)
	print(ps)
	return ps


def main():
	tf.logging.set_verbosity(tf.logging.INFO)

	inputs, outputs = get_data()
	

	print('Inputs: ', len(inputs))
	print('Outputs: ', len(outputs))
	if(len(inputs) < len(outputs)):
		outputs = outputs[:len(inputs)]
	else:
		inputs = inputs[:len(outputs)]

	print('Outputs after length reduction:', len(outputs))
	# print(outputs[250:600])

	## create train and test set
	split = int(len(inputs)*train_split)

	# train_x = inputs
	# train_y = outputs
	train_x = inputs[:split]
	train_y = outputs[:split]


	print(type(train_x))
	print(type(train_y))

	## Instantiate the model
	cnn_classifier = tf.estimator.Estimator(
		model_fn = cnn,
		model_dir = cwd+'/tmp/cnn_model')

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	## Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {'x':train_x},
		y = train_y,
		batch_size = 10,
		num_epochs = None,
		shuffle = True)

	cnn_classifier.train(
		input_fn = train_input_fn,
		steps = 10000,
		hooks=[logging_hook])

	## Test the model

	test_x = inputs[split:]
	test_y = outputs[split:]
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {'x': test_x},
		y = test_y,
		num_epochs = 1,
		shuffle = False)	
	res = cnn_classifier.evaluate(input_fn = test_input_fn)
	print(res)

if __name__=="__main__":
	main()
