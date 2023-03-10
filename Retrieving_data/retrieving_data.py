from os import walk
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from glob2 import glob

from PIL import Image 
from os.path import dirname, realpath, basename
from numpy import asarray, append, array
from matplotlib.pyplot import show, figure, imshow

#-----------------------------------------------------------------------------

def accessing (path, resol) : 

    # retrieving the number of images to be treated in the folder
    # returns an iterator 

    DIR = path
    counter = 0
    for root, dirs, files in walk(DIR) :
        for file in files:    
            if file.endswith('.ppm') or file.endswith('.jpeg'):
                counter += 1

    print("number of images in the folder : ", counter)

    # Creating an image.DirectoryIterator to work over the images of the folder  

    datagen = ImageDataGenerator(rescale=1./255)
    set = datagen.flow_from_directory(path,target_size = resol,
    batch_size = counter,class_mode = 'binary', color_mode='rgb')

    return set

#-----------------------------------------------------------------------------

def store (iter) : 

    # Storing all the information in arrays for convenience 
    # returns arrays 

    X_iter , y_iter = iter.next()

    print("Shape of X : ", X_iter.shape)
    print("Shape of y : ", y_iter.shape)

    return X_iter, y_iter

#-----------------------------------------------------------------------------

def accessing_2 (path) : 

    # Retrieving the names of all files finishing with .ppm or .jpeg in the mentionned Directory 
    # returns a list of string 

    names = glob(path + "\\**\\*.ppm") + glob(path + "\\**\\*.jpeg")
    return names 

#-----------------------------------------------------------------------------

def store_2 (names, X_array, y_array, resol) : 

    # We'll process each image 

    for i in range(len(names)):

        # We store the category 

        # dirname(realpath()) gives the current location of the name that we process 
        # basename gives the name of the upper directory which is the category that we must predict 
        y_array.append((float(basename(dirname(realpath(names[i]))))))

        # We open the image 
        img = Image.open(names[i])
        # We resize the image 
        img = img.resize((resol[1],resol[0]))

        # We store the image 
        img = array(img)

        X_array.append(img)

    y_array = array(y_array)
    X_array =  array(X_array)

    print("Shape of X_train : ", X_array.shape)
    print("Shape of y_train : ", y_array.shape)

    return X_array, y_array