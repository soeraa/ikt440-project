import numpy as np
import cv2 
##crop frame and resize as to get the actual screen value (I think). 
##Code shamelessly copied from: https://github.com/av80r/coaster_racer_coding_challenge/blob/master/train.py
##Is used to train with visual input as state
def processFrame(observation_n):
	if observation_n is not None:
		obs = observation_n[0]['vision']
		obs = cropFrame(obs)
		obs = cv2.resize(obs, (64, 64))
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		obs = obs.astype(np.float32)
		obs *= (1.0 / 255.0)
		obs = np.reshape(obs, [64, 64, 1])
	return obs

def cropFrame(obs):
	return obs[84:564, 18:658, :]

