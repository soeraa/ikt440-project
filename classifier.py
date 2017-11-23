import os
from natsort import natsort_keygen, ns
import numpy as np
import cv2
import utils
import csv

natsort_key1 = natsort_keygen(key=lambda y: y.lower())


def parse_image(img):
	img = cv2.resize(img, (120, 160))
	img = img.astype(np.float32)
	img *= (1.0 / 255.0)
	img = np.reshape(img, [120, 160, 1])


cwd = os.getcwd()
image_path = cwd+'/../recordings/images/mov2/'
image_folder = os.listdir(image_path)
image_folder.sort(key=natsort_key1)

inputs = []
for file in image_folder:
	p = image_path+file
	img = cv2.imread(p, 0)
	inputs.append(parse_image(img))	


key_path = cwd+'/../recordings/keylogs/mov2_parsed.csv'
outputs = []
with open(key_path, 'r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		outputs.append(row)

print('Inputs: ', len(inputs))
print('Outputs: ', len(outputs))
outputs = outputs[:len(inputs)]
print('Outputs after length reduction:', len(outputs))

cv2.imshow('image', inputs[1200])
cv2.waitKey(0)
cv2.destroyAllWindows()
print(outputs[1200])