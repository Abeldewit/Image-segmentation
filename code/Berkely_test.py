#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy.core.fromnumeric import shape
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import skimage.color as color
import skimage.io as io
from scipy.io import loadmat

from plotclusters3D import plotclusters3D
from segment import imSegment, plot_three_imgs
# %%
img = io.imread('./data/181091.jpg')
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
plt.imshow(img)
plt.show()

#%%
labels, peaks, images = imSegment(img, r=20, c=4, pos=True, scale=0.3)

plot_three_imgs(images[0], images[1], images[2])
# %%
