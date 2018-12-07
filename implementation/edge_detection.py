# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('/home/kesavan/Desktop/CTDT/dataset/train/class1/IM29_Frame16.jpg',0)
# # global thresholding
# ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in xrange(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt

def write_otsu_image(img, output_path):
	blur = cv2.GaussianBlur(img, (5, 5), 0)
	ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
	cv2.imwrite(output_path , thresh)