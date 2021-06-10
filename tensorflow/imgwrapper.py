from PIL import Image, ImageShow
from predict import predict
import numpy as np
import models
import cv2



def run_image():
	cap = cv2.VideoCapture(0)
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
		cv2.waitKey(0)
		cv2.destroyAllWindows()