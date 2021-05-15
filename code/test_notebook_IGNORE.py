#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import skimage.color as color
from scipy.io import loadmat

from plotclusters3D import plotclusters3D
from segment import meanshift
from segment import meanshift_opt
from segment import imSegment

#%%
data = loadmat('./data/pts.mat')['data']
#data = color.lab2rgb(data.transpose()).transpose()

labels, peaks = meanshift_opt(data, 2, 4)
# %%
plotclusters3D(data.transpose(), labels, peaks)

# %%

print(data.transpose().shape)
print(labels.shape)
print(peaks.shape)

print((peaks.min(), peaks.max()))
# %%
"""
    Now this is working with the matlab, let's try images
"""

bgr_house = cv2.imread('./data/house.png')
rgb_house = cv2.cvtColor(bgr_house, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_house)
# %%
labels, peaks, flat = imSegment(rgb_house, 80)

#%%
print(flat.shape)
print(labels.shape)
print(peaks.shape)

print((peaks.min(), peaks.max()))

#%%

plt.imshow(rgb_house)

#%%
plt.imshow(labels.reshape(128, 128))

# %%
colors = [c for c in peaks.transpose()]
label_cols = np.array([colors[l] for l in labels]).reshape(128, 128, 3)
label_rgb = cv2.cvtColor(label_cols, cv2.COLOR_LAB2RGB)

plt.imshow(label_rgb)

#%%
from skimage import io

house = io.imread('./data/house.png')
house = color.rgba2rgb(house)
plt.imshow(house)

labels, peaks, flat = imSegment(house, 20)

#%%
plt.imshow(labels.reshape(128, 128))

#%%
seg_im = color.lab2rgb(flat.reshape(128, 128, 3))

plt.imshow(seg_im)

#%%
print(flat.shape, (flat.min(), flat.max()))
print(labels.shape, (labels.min(), labels.max()))
print(peaks.T.shape, (peaks.min(), peaks.max()))

#%%
plotclusters3D(flat, labels, color.lab2rgb(peaks.T).T)