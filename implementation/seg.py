# # import cv2
# # from skimage.filters import frangi
# # img = cv2.imread('/home/kesavan/Desktop/CTDT/collected_images/IM23_Frame16.jpg')

# # obj = frangi(img)
# # cv2.imshow('sample', img)
# # cv2.waitKey(0)

# import os
# from skimage.filters import frangi, hessian
# path = '/home/kesavan/Desktop/CTDT/collected_images/'

# filepath = os.path.join(path, 'Coronary-Angiogram.jpg')
# from skimage import io
# sample = io.imread(filepath)

# for i in dir(io):
# 	print i
# sample = frangi(sample)

# print(sample)
# t = io.imshow(sample, plugin = None)
# io.show()


from skimage.data import camera
from skimage.filters import frangi, hessian

import matplotlib.pyplot as plt
from skimage import io


import cv2

image = cv2.imread('/home/kesavan/Desktop/hop_dataset/DS46/S1/IMG00000_Frame26.jpg', 0)

fig, ax = plt.subplots(ncols=1)

# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title('Original image')

ax.imshow(frangi(image), cmap=plt.cm.gray)

# ax[2].imshow(hessian(image), cmap=plt.cm.gray)
# ax[2].set_title('Hybrid Hessian filter result')


ax.axis('off')

plt.tight_layout()
plt.show()

fig.savefig('/home/kesavan/Desktop/sample.jpg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)