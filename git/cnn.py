#Import all the nessesary libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import os
from os import listdir

#Set caffe to cpu or gpu mode, either is caffe.set_mode_cpu() or caffe.set_mode_gpu()
caffe.set_mode_cpu()

#Define variable for location of required files
MODEL_FILE = '/home/foo/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/home/foo/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN_FILE = '/home/foo/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
LABEL_FILE = '/home/foo/caffe/data/ilsvrc12/synset_words.txt'

#Load the BVLC Reference Caffenet models
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
'''net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_channel_swap('data',(2,1,0))
net.set_raw_scale('data',255)
net.set_mean('data',np.load(MEAN_FILE))'''

pred_features = []

for f in sorted(listdir('/home/foo/Project/Dataset/')):
	#Convert image to caffe inputs
	IMAGE_FILE = f
	input_image = caffe.io.load_image('/home/foo/Project/Dataset/'+f)
	
	'''
	#Let plot out the image
	plt.imshow(input_image)
	plt.savefig('/home/cat.png')
	plt.close()
	'''

	#Predict class
	prediction = net.predict([input_image])
#	print 'prediction shape:', prediction[0].shape[0]
#	print 'predicted class:', prediction[0].argmax()

	#Predict label
	fi = open(LABEL_FILE)
	labels = fi.readlines()
#	print 'predicted name:', labels[prediction[0].argmax()],
	print f,':', labels[prediction[0].argmax()],

	fit=open('/home/foo/Project/predicted_labels.txt','a')
	fit.write(f+':'+str(labels[prediction[0].argmax()]))
	fit.close()

	
'''replace prediction[0].argmax() with class value like 281'''
'''
#Plot the polygon frequency
plt.plot(prediction[0])
plt.savefig('/home/prediction.png')
plt.close()
'''
