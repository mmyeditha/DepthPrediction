from PIL import Image, ImageShow
from matplotlib import pyplot as plt
from predict import predict
import numpy as np
import time
import models
import cv2



def plot_cv():
	"""
	Starts video capture session and plots images with
	open-cv. Some data is lost.
	"""
	cap = cv2.VideoCapture(0)
	print("Press a for next image, q to quit")

	while(True):

		ret, frame = cap.read()
		# return frame
		img = Image.fromarray(frame)
		pred = predict('models/NYU_FCRN.ckpt', img)[0]
		new_pred = np.zeros((128, 160), np.uint8)
		max = np.max(pred)
		min = np.min(pred)

		for i in range(0, 128):
			for j in range(0, 160):
				new_a = pred[i][j][0]- min
				new_pred[i][j] = (new_a/max)*255

		cv2.imshow('img', new_pred)
		if cv2.waitKey(33) == ord('a'):
			cv2.destroyAllWindows()
		elif cv2.waitKey(33) == ord('q'):
			cv2.destroyAllWindows()
			break

def plot_mpl():
	"""
	Plots in matplotlib, this is the original implementation
	"""
	cap = cv2.VideoCapture(0)
	print("Press a for next image, q to quit")

	while(True):

		_, frame = cap.read()
		
		# return frame
		img = Image.fromarray(frame)
		pred = predict('models/NYU_FCRN.ckpt', img)
		fig = plt.figure()
		ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
		fig.colorbar(ii)
		plt.show()
		time.sleep(0.5)
		plt.close('all')

		inp = input('Next? (y/n)')
		if inp == 'y':
			continue
		elif inp == 'n':
			break


plot_mpl()
