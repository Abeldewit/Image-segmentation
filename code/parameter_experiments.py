#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import skimage.color as color
import skimage.io as io
import time
import pickle

from plotclusters3D import plotclusters3D
from segment import imSegment, plot_three_imgs
# %%
img = io.imread('../data/face.jpg')
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
plt.imshow(img)
plt.show(block=False)
    
# %%
scales = [0.25, 0.5, 1]
r_values = [2, 5, 10, 20, 50]
c_values = [2, 4, 8, 10, 20]
feature_types = [False, True]
#%%
# Testing different 'r'
r, c, s, f = 30, 4, 0.3, False
for r in r_values:
    tic = time.perf_counter()
    labels, peaks, imgs = imSegment(img, r=r, c=c, pos=f, scale=s, experiment=True)
    toc = time.perf_counter()
    print('\n\n',toc-tic)

    plt.imshow(imgs[2])
    plt.suptitle('(r:{}, c:{}, s:{}, f:{})'.format(r, c, s, f))
    plt.title(f'Time needed: {toc-tic:.4f}')
    plt.savefig(f'../data/output/r_test_{r}.jpg')
    plt.show()
    
# %%
# Testing different 'c'
r, c, s, f = 10, 4, 0.3, False
for c in c_values:
    tic = time.perf_counter()
    labels, peaks, imgs = imSegment(img, r=r, c=c, pos=f, scale=s, experiment=True)
    toc = time.perf_counter()
    print('\n\n',toc-tic)
    
    plt.imshow(imgs[2])
    #plt.show()
    plt.suptitle('(r:{}, c:{}, s:{}, f:{})'.format(r, c, s, f))
    plt.title(f'Time needed: {toc-tic:.4f}')
    plt.savefig(f'../data/output/c_test_{c}.jpg')
    plt.close()
# %%
