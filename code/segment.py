import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage import color
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import cv2
from plotclusters3D import plotclusters3D

def findpeak(data, idx, r):
    """
    Function to find the mean-shift peak for a certain point in the data
    Args:
        data:   the n-dimensional dataset consisting of p points
        idx:    the column index of the data point for which we wish to compute its associated density peak
        r:      the search window radius.
    """
    point = data[:, idx]
    
    shift = 10000
    t = 0.01
    while shift > t:
        # Find points that are within 'r' to the current point
        inside = []
        # for idy in range(data.shape[1]):
        #     other = data[:, idy]

        #     dist = norm(point - other)
        #     if dist <= r:
        #         inside.append(other)
        
        p = point.reshape(1, point.shape[0])
        d = data.transpose()
        distances = cdist(p, d, metric='euclidean')
        
        indices = np.where(distances <= r)[1]
        inside = d[indices]

        # Calculate the mean of all 'close' features
        new_p = inside.mean(axis=0)
        
        shift = norm(point - new_p)
        point = new_p
        
    # So we found the peak for the point at idx
    return point


def meanshift(data, r):
    labels = np.zeros(shape=data.shape[1])
    peaks = []

    # We iterate over each data point
    for idx in tqdm(range(data.shape[1])):
        # We calculate the peak for this datapoint
        new_peak = findpeak(data, idx, r)
        label = len(peaks)

        # We check whether the found peak corresponds to an existing peak
        new = True
        for l, peak in enumerate(peaks):
            if norm(peak - new_peak) < (r/2):
                label = l
                new = False

        if new:
            peaks.append(new_peak)
        
        labels[idx] = label
    

    labels = labels.reshape(1, labels.shape[0])
    peaks = np.array(peaks)
    peaks = peaks.reshape(peaks.shape[1], peaks.shape[0])
    return labels, peaks


def findpeak_opt(data, idx, r, c):
    """
    Function to find the mean-shift peak for a certain point in the data.
    The optimized version of findpeak using:
        - Basin of attraction: saving the datapoints that are close to the initial datapoint
        - Search-path: savind the datapoints that are encountered during the 'walk' of the mean-shift
    Args:
        data:   the n-dimensional dataset consisting of p points
        idx:    the column index of the data point for which we wish to compute its associated density peak
        r:      the search window radius.
    """
    point = data[:, idx]
    
    shift = 10000
    t = 0.01

    first_it = True
    neighbors = None

    search_path_neighbors = None
    while shift > t:
        # Find points that are within 'r' to the current point
        inside = []
        
        # Calculate all distances to the current point
        p = point.reshape(1, point.shape[0])
        d = data.transpose()
        distances = cdist(p, d, metric='euclidean')
        
        # And save the indexes of points within the search window
        indices = np.where(distances <= r)[1]
        inside = d[indices]

        # Save the neighbors of current data point for the basin
        if first_it:
            neighbors = indices
        
        # Save a list of all points that are within r/c of our search path
        sp_inx = np.where(distances <= (r/c))[1]
        search_path_neighbors = np.unique(np.append(search_path_neighbors, sp_inx)) if search_path_neighbors is not None else sp_inx

        # Calculate the mean of all 'close' features
        new_p = inside.mean(axis=0)
        
        shift = norm(point - new_p)
        point = new_p
        
    # So we found the peak for the point at idx
    return point, neighbors, search_path_neighbors


def meanshift_opt(data, r, c):
    """
    Finds the peaks for all datapoints using the mean-shift algorithm
    Improvement of the 'meanshift' function using the basin of attraction 
    and the search path. These datapoints will get the same label as the datapoint
    that the search was started from, unless they are already labeled. 
    Args:
        data: image data in the format
        r: size of the window
        c: size of the search_path window

    Returns:
        labels: an array of length [number of pixels] with a label for each pixel
        peaks: an array of size [number of pixels]x[feature vector]
    """
    # For the basin of attraction, we check if a point has a label already 
    # by filling the array with negative values
    labels = np.full(shape=data.shape[1], fill_value=-1)
    peaks = []

    # We iterate over each data point
    for idx in tqdm(range(data.shape[1])):
        if labels[idx] == -1:
            # We calculate the peak for this datapoint
            new_peak, basin, search_path = findpeak_opt(data, idx, r, c)
            label = len(peaks)

            # We check whether the found peak corresponds to an existing peak
            #TODO Use cdist here as well
            new = True
            for l, peak in enumerate(peaks):
                if norm(peak - new_peak) < (r/2):
                    # If this is not a new peak, the label will be the same as the
                    # exisiting peak
                    label = l
                    new = False
            
            # If this is a new peak, it's added to the list
            if new:
                peaks.append(new_peak)
            
            labels[idx] = label

            # Basin & search path are the datapoints in the basin and search path
            l = labels.transpose()
            
            # These lists of indices are merged and only unique indices are kept
            neighbors = np.unique(np.append(basin, search_path))

            # With these indices, a check can be performed whether they have a label
            # and if not, assign the current label
            optimized_labels = np.where(l[neighbors] == -1, label, l[neighbors])
            l[neighbors] = optimized_labels

            labels = l.transpose()
        
    

    labels = labels.transpose()
    peaks = np.array(peaks)
    #peaks = peaks.reshape(peaks.shape[1], peaks.shape[0])
    return labels, peaks


def imSegment(img, r, c=4, pos=False, lab=True, scale=0.5, blur=3, experiment=False, opt=True):
    """
    Takes in an RGB image, converts it to CIELAB and creates clusters
    Returns four images:
        1. The original image
        2. The prepocessed image
        3. The segmented image
        4. The 3D plot

    Args:
        img: Image within the RGB colorspace
        r: Mean-shift window size
    """
    og_image = img.copy()
    pr_image = None
    se_image = None
    td_image = None

    # Do some preprocessing
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    # Apply preprocessing such as scaling and blurring
    blurred = cv2.GaussianBlur(img, (blur,blur), cv2.BORDER_DEFAULT)
    img = cv2.resize(blurred, dim, interpolation=cv2.INTER_AREA)
    pr_image = img.copy()

    # Determine preprocessing such as color space and positional encoding
    if lab:
        lab_image = color.rgb2lab(img)
    
    if pos:
        lab_image = __add_position(lab_image)

    # Turn the 3D image into an array of vectors
    flat_image = lab_image.reshape(-1, lab_image.shape[2])

    # Run the Mean-shift algorithm
    if opt:
        labels, peaks = meanshift_opt(flat_image.transpose(), r, c)
    else:
        labels, peaks = meanshift(flat_image.transpose(), r)
    
    # Create an image 
    peak_colors = color.lab2rgb(peaks[:, :3])
    seg = np.array([peak_colors[l] for l in labels], dtype=float)
    se_image = seg.reshape(img.shape[0], img.shape[1], img.shape[2])

    if not experiment:
        peak_colors = color.lab2rgb(peaks[:, :3])*255
        peak_colors[:, [0, 1, 2]] = peak_colors[:, [2, 1, 0]]
        plotclusters3D(flat_image, labels, peak_colors)
        plt.show(block=False)

    return labels, peaks, (og_image, pr_image, se_image)


def __add_position(img):
    
    new_img = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]+2))

    for x, y in itertools.product(range(img.shape[0]), range(img.shape[1])):
        pixel = img[x, y]
        pos = [x, y]

        new_vec = np.array([pixel[0], pixel[1], pixel[2], pos[0], pos[1]])
        new_img[x, y] = new_vec
    return new_img


def plot_three_imgs(one, two, three, title=''):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    

    axs[0].imshow(one)
    axs[0].axis('off')
    axs[0].title.set_text('Original Image')

    axs[1].imshow(cv2.cvtColor(two, cv2.COLOR_Lab2RGB))
    axs[1].axis('off')
    axs[1].title.set_text('Processed Image (in LAB)')

    axs[2].imshow(three)
    axs[2].axis('off')
    axs[2].title.set_text('Segmented Image')

    fig.tight_layout()

    if title != '':
        fig.suptitle(title)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.85, 
                        wspace=0,
                        hspace=0)
    plt.show()