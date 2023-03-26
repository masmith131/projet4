from os import listdir, walk
from numpy import asarray, append, array
from PIL import Image
from csv import DictWriter
from keras.preprocessing.image import ImageDataGenerator


def store(path, resol) : 

    # We store all of the images from the kaggle folder in an array 
    # recall that target has been defined above 

    images = [] 
    names = []

    # get the path/directory
    folder_dir = path

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm")):
            img = Image.open(folder_dir + '/' + image)
            img = img.resize(resol) # (30,30) as an example 
            img = asarray(img)
            images.append(img) 
            names.append(image.replace('.ppm',''))

    images = array(images)

    print("Number of images and their resolution in the kaggle dataset : ", images.shape)

    return images, names

#-----------------------------------------------------------------------------

def write(names, predictions, title) : 

    # Here is the code to write the results in a CSV for kaggle named title.csv

    with open(title + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Category']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(predictions)): 
            writer.writerow({'Id' : names[i], 'Category' : predictions[i]})

#-----------------------------------------------------------------------------

from keras.utils import load_img
from keras.utils import img_to_array

import cv2 

def store_2(path, resol) : 

    # We store all of the images from the kaggle folder in an array 
    # recall that target has been defined above 

    images = [] 
    names = []

    # get the path/directory
    folder_dir = path

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm") or image.endswith(".jpeg")):
            img = cv2.imread(folder_dir + '/' + image)
            RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized=cv2.resize(RGB, (resol[0],resol[1]))
            arr = asarray(resized)
            images.append(arr) 
            names.append(image.replace('.ppm',''))

    images = array(images)

    print("Number of images and their resolution in the kaggle dataset : ", images.shape)

    return images, names