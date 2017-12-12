import gym
import universe  # register the universe environments
import logging
import random
import numpy as np
import cv2
import utils
import tensorflow as tf
<<<<<<< HEAD
import os
=======
>>>>>>> 31622488e152602986878286eef58bce26edb035
from model import cnn
from classifier import classify

print('Loading..')

env_id = 'flashgames.CoasterRacer-v0'

env = gym.make(env_id)
env.configure(fps=5, remotes=1)


## set environment variables
speed_up = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowUp', True)]
left = [('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowUp', True)]
right = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowUp', True)]

actions = [speed_up, right, left]
epsilon = 0.6
gamma = 0.99
gt = 1
exploring = 0


##some init action
observation_n = env.reset()
action_n = [speed_up for ob in observation_n]
s = 0


## init tensorflow

while True:
	while(observation_n[0] is None):
		action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
		observation_n, reward_n, done_n, info = env.step(action_n)

	observation = utils.processFrame(observation_n)

	predictions = classify(observation)

	index = predictions[0]['classes']
	print('prediction action is: ',index)
	action_n = [actions[index] for ob in observation_n]

	observation_n, reward_n, done_n, info = env.step(action_n)

	if(reward_n[0] > 0.0):
		# if positive reward we will reinforce and encourage learning
		print("good choice")
	else:
		print("bad choice")
	print("Selected choice gave a reward of {}".format(reward_n[0]))



