import cv2
import os
import edge_detection

input_path = '/home/kesavan/Desktop/amarula/S2/'
output_path = '/home/kesavan/Desktop/'

def load_images(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename), 0)
		if img is not None:
			images.append(img)
	return images

def main():
	images = load_images(input_path)
	count = 1;
	for i in images:
		edge_detection.write_otsu_image(i, output_path + str(count) + '.jpg')
		count = count + 1

if __name__ == '__main__':
	main()

# import numpy as np
# import scipy.signal
# import matplotlib.pyplot as plt
# from skimage import io, color
# from skimage import exposure
# img = io.imread('/home/kesavan/Desktop/sample.jpg')    # Load the image
# img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)
# kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# # we use 'valid' which means we do not add zero padding to our image
# edges = scipy.signal.convolve2d(img, kernel, 'valid')
# # print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255
# # Adjust the contrast of the filtered image by applying Histogram Equalization
# edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)
# plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
# plt.axis('off')
# plt.show()