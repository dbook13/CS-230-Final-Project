# Separate each video into a sequence of images

import subprocess as sp
import os

path = 'Videos'

for filename in os.listdir(path):
	if filename != '.DS_Store':
		path_in = 'Videos/' + filename
		name = filename[:len(filename)-4]
		path_out = 'Sequences/' + name + '%03d.jpg'
		cmd = 'ffmpeg -i ' + path_in + ' ' + path_out
		print(cmd)
		sp.call(cmd,shell=True)