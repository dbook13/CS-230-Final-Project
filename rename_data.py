# rename every image that ends in d1 or g1 to d11 or g11 respectively

import os

for filename in os.listdir('SequencesOutput2'):
	name = filename[:len(filename)-18]
	if(name[-1] == '1'):
		cur = name + '1' + filename[len(filename)-18:]
		os.rename('SequencesOutput2/'+filename,'SequencesOutput2/'+cur)