import numpy as np
from PIL import Image
import os
import cv2

def image_to_array(img_path):
    """
    Returns a numpy array of pixels of the image.
    """
    # load image from file and convert to grayscale
    img = cv2.imread(img_path, 0)
    # scale image to be 0 and 1
    new_img = img / 255.0 # now all values are ranging from 0 to 1
    # convert to array
    pixels = np.asarray(new_img)
    return pixels

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def create_data(file_path):
    """
    Given file path to folder of data, iterate through all folders/subfolders and extract each png.
    Create a numpy array of the grayscale pixels of each png and pad them so that they are all the same size.
    """
    # file path to training dataset eg. r'/Users/admin'
    directory = file_path

#     max_rows, max_columns = data_shapes.max(axis=0) <-- data_shapes = array of pixels.shape
    max_rows, max_columns = 154, 158
    
    data = []
    labels_pitch = []
    labels_duration = []
#     culprits = []
    for note in os.listdir(directory):
        if note == '.DS_Store' or note == 'combined_dataset':
            continue
            
        for note_type in os.listdir(directory+'/'+note):
            if note_type == '.DS_Store':
                continue
            
            for i in os.listdir(directory+'/'+note+'/'+note_type):
                if i == '.DS_Store':
                    continue
                    
                # add to labels
                all_pitches = ['c1', 'd1', 'e1', 'f1', 'g1', 'a1', 'b1', 'c2', 'd2', 'e2', 'f2', 'g2', 'a2', 'rests']
                labels_pitch.append(all_pitches.index(note))

                d = 0
                if note_type == 'whole':
                    d = 0
                elif note_type == 'half_dotted':
                    d = 1
                elif note_type == 'half':
                    d = 2
                elif note_type == 'quarter_dotted':
                    d = 3
                elif note_type == 'quarter':
                    d = 4
                elif 'eighth_dotted' in note_type:
                    d = 5
                elif 'eighth' in note_type:
                    d = 6
                elif 'sixteenth'in note_type:
                    d = 7
                    
                labels_duration.append(d)

                # retrieve image pixels from path
                img_path = directory+'/'+note+'/'+note_type+'/'+i
                pixels = image_to_array(img_path)
                
#                 if pixels.shape[0] == 154 or pixels.shape[1] == 158:
#                     culprits.append(pixels)
                
                # figure out which boundary would the image hit first if resized
                # then resize accordingly
                if pixels.shape[0]/max_rows > pixels.shape[1]/max_columns:
                    pixels = image_resize(pixels, height=max_rows)
                    
                    to_add = max_columns-pixels.shape[1]
                    add_to_right = to_add//2
                    add_to_left = to_add - add_to_right
                    
                    pixels = np.hstack((pixels, np.ones((pixels.shape[0], add_to_right))))
                    pixels = np.hstack((np.ones((pixels.shape[0], add_to_left)), pixels))
                else:
                    pixels = image_resize(pixels, width=max_columns)
                    
                    to_add = max_rows-pixels.shape[0]
                    add_to_top = to_add//2
                    add_to_bottom = to_add - add_to_top
                    
                    pixels = np.vstack((pixels, np.ones((add_to_bottom, pixels.shape[1]))))
                    pixels = np.vstack((np.ones((add_to_top, pixels.shape[1])), pixels))
                
                # add resized and padded image pixels to data
                data.append(pixels)

#     return np.array(data), np.array(culprits)
    return np.array(data), np.array(labels_pitch), np.array(labels_duration)

"""
code to test function:

data, labels_pitch, labels_duration = create_data(r"/Users/claricewang/CogWorks/bwsiweek4/SheetMusicCV/note_dataset_digital")
fig, ax = plt.subplots()
ax.imshow(data[500], cmap='Greys_r')
                ^ this can be any int [0,560]
"""
