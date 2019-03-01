import cv2
import numpy as np
def split(img,n):
	# return array of blocks of size 3n_square
	width = len(img[0])
	height = len(img)
	result = []
	temp = []
	counter = 0
	for i in range(height):
		for j in range(width):
			temp.append(img[i][j][0])
			temp.append(img[i][j][1])
			temp.append(img[i][j][2])
			counter+=1
			if ( counter == 3*(n**2) ):
				result.append(temp)
				counter = 0
				temp = []
	return np.array(result)

def join(C,n,width,height):
	result_image = np.zeros((height,width,3))
	result_image = result_image.tolist()
	counter = 0
	curr_row = 0
	for i in range(height):
		for j in range(width):
			result_image[i][j] = [ C[curr_row][counter],C[curr_row][counter+1],C[curr_row][counter+2] ]
			counter+=1
			if ( counter == 3*(n**2) ):
				counter = 0
				curr_row +=1
	return np.array(result_image)
