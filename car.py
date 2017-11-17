import gym
import universe  # register the universe environments
import logging
import random
import numpy as np
import cv2

env_id = 'flashgames.CoasterRacer-v0'

env = gym.make(env_id)
env.configure(remotes=1)

## get environment variables
reg = universe.runtime_spec('flashgames').server_registry
height = reg[env_id]["height"]
width = reg[env_id]["width"]

## set environment variables
speed_up = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowUp', True)]
left = [('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowUp', True)]
right = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowUp', True)]
actions = [left, right]


##some init action
action_n = [speed_up for ob in observation_n]
observation_n = env.reset()
s = 0


##crop frame and resize as to get the actual screen value (I think). 
##Code shamelessly copied from: https://github.com/av80r/coaster_racer_coding_challenge/blob/master/train.py
##Is used to train with visual input as state
def cropFrame(observation):
    # adds top = 84 and left = 18 to height and width:
	obs = observation[0]['vision']
	obs = obs[84:84+height, 18:18+width]
	obs = cv2.resize(obs, (120, 160))
	    # greyscale
	obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
	obs = obs.astype(np.float32)
	# scale from 1 to 255
	obs *= (1.0 / 255.0)
	# re-shape a bitch
	obs = np.reshape(obs, [120, 160])
	return obs


while True:
	while(observation_n[0] is None):
		action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
		observation_n, reward_n, done_n, info = env.step(action_n)


	action_n = [random.choice(actions) for ob in observation_n]
	observation_n = cropFrame(observation_n)
	s += 1
	observation_n, reward_n, done_n, info = env.step(action_n)


	# env.render()
	 
