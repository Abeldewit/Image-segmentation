import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import sys, getopt
import skimage.color as color
import cv2


sys.path.append('./code/')
from plotclusters3D import plotclusters3D
from segment import imSegment, plot_three_imgs

help_text = "\
\n*** Image Segmentation Script ***\n \
\n\
The following options are available: \n\n\
-i (or --image)\t\t The desired input image to perform segmentation on\n\
-r \t\t\t The search window that will be used by the Mean-Shift algorithm \n\
-f (or --feature_type)\t The feature type that is used, either '3D' (pixel data) or '5D' including positional encoding \n\
-s (or --scale) \t A scalar value between 0 and 1 so scale down the image for faster processing\n\n\
"

def main(argv):
    input_file = ''
    r = 2
    s = 1
    feature_type = '3D'

    # Reading in the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hi:r:f:s:", ["image=", "r=", "feature_type=", "scale="])
    except getopt.GetoptError:
        print('imageSegmentation.py -i <input image> -r <search_window> -f <feature_type>')
        sys.exit(2)
    
    if len(opts) == 0:
        print("No input provided!")

    # Processing the cli arguments 
    for opt, arg in opts:
        if opt == '-h':
            print(help_text)
            sys.exit(1)
        if opt in ('-i', '--image'):
            input_file = arg
        if opt in ('-r', '--r'):
            try:
                r = int(arg)
            except ValueError as e:
                print('Search window has to be an integer')
                sys.exit(2)
        if opt in ('-f', '--feature_type'):
            if arg in ('3D', '5D'):
                feature_type = arg
            else:
                print("Feature type has to be either '3D' or '5D'")
                sys.exit(2)
        if opt in ('-s', '--scale'):
            try:
                s = float(arg)
                if not 0 < s <= 1:
                    raise ValueError("Scale factor has to be between 0 and 1")
            except ValueError as e:
                print(e)
                sys.exit(2)

    # Reading in the image file
    try:
        image = io.imread(input_file)
    except FileNotFoundError:
        print("The image file can not be found!")
        print(input_file)
        sys.exit(2)

    # If the image contains an alpha channel, remove it
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    p = True if feature_type == '5D' else False

    labels, peaks, images = imSegment(image, r, c=4, pos=p, scale=s)
    plot_three_imgs(images[0], images[1], images[2])
    plt.show()
    

    

if __name__ == "__main__":
    main(sys.argv[1:])