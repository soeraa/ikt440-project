import cv2
import os


data_path = '/home/aasmund/Documents/universe/recordings/images/'
parsed_path = '/home/aasmund/Documents/universe/recordings/parsed/'


direction_folders = os.listdir(data_path)
p_direction_folders = os.listdir(parsed_path)
print('parsing images')
for direction in direction_folders:
	images_path = os.path.join(data_path, direction)
	p_images_path = os.path.join(parsed_path, direction)
	s=0
	# print(direction)
	for image in os.listdir(images_path):
		image_path = os.path.join(images_path, image)
		image_name = str(s)+'-'+direction+'.png'
		p_image_path = os.path.join(p_images_path, image_name)
		# 0 entails grayscale image
		img = cv2.imread(image_path, 0)
		print(img.shape)
		img = cv2.resize(img, (64, 64))
		cv2.imwrite(p_image_path, img)
		# print(p_image_path)
		s += 1
		if(s%100 == 0):
			print(s)




