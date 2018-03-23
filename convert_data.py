# Convert json data to numpy array

import json
import numpy as np
import os
from sklearn.utils import shuffle

words = ['awful','cat','dirty','house','paper','remember','teacher','thank you','want'
		'yellow']

end_sign = [57,46,48,50,61,60,58,63,56,57,60,50,57,48,46,48,49,51,56,51,51,58,61,47,
61,53,54,43,52,61,80,71,68,73,63,70,68,71,83,74,61,60,65,66,76,78,83,70,69,64,72,75,
76,81,62,69,79,69,60,73,50,58,60,54,50,68,52,52,58,58,69,69,65,65,66,65,71,63,72,64,
63,50,52,53,61,56,59,45,57,43,75,66,70,73,71,82,90,75,82,83,61,61,54,52,66,48,51,45,
60,56,57,52,52,45,55,58,66,70,65,59,56,45,65,64,70,77,68,116,73,62,90,83,87,84,90,89,
93,90,85,97,60,55,56,60,57,57,59,54,59,56,71,55,54,60,64,59,45,64,74,75,74,46,46,45,
48,39,51,46,47,44,55,59,63,53,53,55,45,59,60,50,51,97,65,84,83,81,76,90,74,93,80,85,
97,78,95,91,86,88,86,78]


num_vids = 200
max_len = 150
X = np.zeros((num_vids,max_len,390))
Y = np.zeros((num_vids,10))


filenames = []

for filename in os.listdir('SequencesOutput2'):
	filenames.append(filename)

filenames = sorted(filenames)


vid_name = ''
count_vid = -1
count_pic = 0

for f in filenames:
	pic_name = f[:len(f)-18]

	if vid_name != pic_name:
		vid_name = pic_name
		count_vid += 1
		count_pic = 0
		with open('SequencesOutput2/' + f) as json_data:
			d = json.load(json_data)

		data = []
		data.extend(d['people'][0]['pose_keypoints'])
		data.extend(d['people'][0]['face_keypoints'])
		data.extend(d['people'][0]['hand_left_keypoints'])
		data.extend(d['people'][0]['hand_right_keypoints'])
		data = np.array(data)
		X[count_vid,count_pic,:] = data
		count_pic+=1

	elif pic_name == vid_name:
		with open('SequencesOutput2/' + f) as json_data:
			d = json.load(json_data)

		data = []
		data.extend(d['people'][0]['pose_keypoints'])
		data.extend(d['people'][0]['face_keypoints'])
		data.extend(d['people'][0]['hand_left_keypoints'])
		data.extend(d['people'][0]['hand_right_keypoints'])
		data = np.array(data)
		if count_pic < 150:
			X[count_vid,count_pic,:] = data
		count_pic+=1


for i in range(len(end_sign)):
	Y[i,:end_sign[i]-1,0] = 1
	Y[i,end_sign[i]-1:end_sign[i]+14,int(np.floor(i/20))+1] = 1
	Y[i,end_sign[i]+14:,0] = 1




X,Y = shuffle(X,Y,random_state = 0)

X_train = X[:160]
X_test = X[160:]
Y_train = Y[:160]
Y_test = Y[160:]

np.save('NPdata/xtrain', X_train)
np.save('NPdata/xtest', X_test)
np.save('NPdata/ytrain', Y_train)
np.save('NPdata/ytest', Y_test)










