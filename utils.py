import numpy as np
import cv2 
##crop frame and resize as to get the actual screen value (I think). 
##Code shamelessly copied from: https://github.com/av80r/coaster_racer_coding_challenge/blob/master/train.py
##Is used to train with visual input as state
def processFrame(observation_n, model_type):


	if observation_n is not None:
		obs = observation_n[0]['vision']
		obs = cropFrame(obs)
		obs = cv2.resize(obs, (64, 64))
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		if(model_type=='canny'):
			obs = cv2.Canny(obs, threshold1 = 200, threshold2 = 400)
		obs = obs.astype(np.float32)
		obs *= (1.0 / 255.0)
		obs = np.reshape(obs, [64, 64, 1])
	return obs

def cropFrame(obs):
	return obs[114:534, 38:638, :]


def parse_image(img):
	# img = cv2.resize(img, (64, 64))
	img = img.astype(np.float32)
	img *= (1.0 / 255.0)
	img = np.reshape(img, [64, 64, 1])
	return img


def parse_canny_image(img):
	e_img = cv2.Canny(img, threshold1 = 200, threshold2=400)
	img = e_img.astype(np.float32)
	img *= (1.0 / 255.0)
	img = np.reshape(img, [64, 64, 1])
	return img