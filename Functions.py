from numpy import zeros, arange, array, asarray
from matplotlib.pyplot import subplots, title, xticks, legend, show, figure, plot, xlabel, ylabel, subplot, tight_layout
from os import listdir, walk
from PIL.Image import open
from sklearn.metrics import classification_report
from csv import DictWriter
from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize, GaussianBlur
from numpy.random import normal
from keras.preprocessing.image import ImageDataGenerator
from glob2 import glob
from os.path import dirname, realpath, basename

#-----------------------------------------------------------------------------

def number (nbr_class, y, name) :     

    # Print the number of signs of each type in the sets

    nbr = zeros(nbr_class, dtype=int)

    for i in range(nbr_class) :
        nbr[i] = int((y.copy() == i).sum())  # Number of images of class i in the set 

    print("Number of each sign in the set : " + name)
    print()
    print(nbr)
    print()
    print("Total of signs : ", nbr.sum())
    print()

    return nbr

#-----------------------------------------------------------------------------

def graphs (nbr_class, set_1, set_2, name_1, name_2) : 

    # We plot an histo showing how many signs of each class we have in each set 

    fig, ax = subplots(figsize = (20, 7))
    bins = [x + 0.5 for x in range(-1, nbr_class)]
    ax.hist([set_1, set_2], range = (0, nbr_class - 1), bins=bins, edgecolor = 'white', color = ['blueviolet','black'], label = [name_1, name_2])
    title("Visualisation of the number of signs of each class in each set")
    xticks(arange(nbr_class))
    legend()
    show()

#-----------------------------------------------------------------------------

def to_jpeg (folder_dir, dest) : 

    # Here is a code to save all ppm in jpeg in a directory (must be created)

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm")):
            img = open(folder_dir + '/' + image)
            img.save(dest + '/' + image.replace('.ppm','.jpg'), format = 'JPEG') 

#-----------------------------------------------------------------------------

def perf(anc) : 

    # Plot of performances of the model on the training sets 

    figure()

    subplot(2, 1, 1)
    plot(anc.history['accuracy'], label='training accuracy', color = 'darkblue')
    plot(anc.history['val_accuracy'], label='test accuracy', color = 'magenta')
    title('Accuracy')
    xlabel('epochs')
    ylabel('accuracy')
    legend()

    subplot(2, 1, 2)
    plot(anc.history['loss'], label='training loss', color = 'darkblue')
    plot(anc.history['val_loss'], label='test loss', color = 'magenta')
    title('Loss')
    xlabel('epochs')
    ylabel('loss')
    legend()

    tight_layout()
    show()

#-----------------------------------------------------------------------------

def ratio_kaggle (y_test_tc, X_test, model) :

    # Here's a function that will give the score that we can see on kaggle based on the testing dataset 

    true = y_test_tc.argmax(axis=1)

    print("True codes : ", true)
    print("Number of true codes : ", len(true))

    predict = model.predict(X_test).argmax(axis=1)

    print("Predictions : ", predict)
    print("Number of predictions : ", len(predict))

    right = 0 

    for i in range(len(true)) : 
        if predict[i] == true[i] :  
            right += 1 

    print("Number of right : ", right)
    print("Number of elements : ", len(true))

    print("Ratio : ", right/len(true))

#-----------------------------------------------------------------------------

def network (model, y_test_tc, X_test, label_names, nbr_class) : 

    # Evaluate the network
    # This displays the precision and recall on each type of sign 

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test) 
    print(classification_report(y_test_tc.copy().argmax(axis=1),
        predictions.argmax(axis=1), target_names=label_names, labels=range(nbr_class)))

#-----------------------------------------------------------------------------

def write(names, predictions, title) : 

    # Here is the code to write the results in a CSV for kaggle named "title.csv"

    with open(title + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Category']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(predictions)): 
            writer.writerow({'Id' : names[i], 'Category' : predictions[i]})

#-----------------------------------------------------------------------------

def Store(path, resol) : 

    # We store all of the images from the kaggle folder in an array using PIL package 
    # recall that target has been defined above 

    images = [] 
    names = []

    # get the path/directory
    folder_dir = path

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm")):
            img = open(folder_dir + '/' + image)
            img = img.resize(resol) # (30,30) as an example 
            img = asarray(img)
            images.append(img) 
            names.append(image.replace('.ppm',''))

    images = array(images)

    print("Number of images and their resolution in the kaggle dataset : ", images.shape)

    return images, names

#-----------------------------------------------------------------------------

def Store_2(path, resol) : 

    # We store all of the images from the kaggle folder in an array using cv2 
    # recall that target has been defined above 

    images = [] 
    names = []

    # get the path/directory
    folder_dir = path

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm")):
            img = imread(folder_dir + '/' + image)
            RGB = cvtColor(img, COLOR_BGR2RGB) # cv2 doesn't have the same convention RGB so we must do this change 
            resized = resize(RGB, (resol,resol))
            arr = asarray(resized)
            images.append(arr) 
            names.append(image.replace('.ppm',''))

    images = array(images)

    print("Number of images and their resolution in the kaggle dataset : ", images.shape)

    return images, names

#-----------------------------------------------------------------------------

def blur (img, resol, sigma) : 

    # This fct applies blur on an image by applying GaussianBlur 

    blur = GaussianBlur(img,(resol+1,resol+1), sigma)

    return blur 

#-----------------------------------------------------------------------------

def noise (img, mean, var) : 

    # This fct applies random (normal) noise on an image 

    row,col,ch= img.shape

    sigma = var**0.5

    gauss = normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noise = img + gauss

    return noise 

#-----------------------------------------------------------------------------

def accessing (path, resol) : 

    # retrieving the number of images to be treated in the folder
    # returns an iterator 

    DIR = path
    counter = 0
    for root, dirs, files in walk(DIR) :
        for file in files:    
            if file.endswith('.ppm') :
                counter += 1

    print("number of images in the folder : ", counter)

    # Creating an image.DirectoryIterator to work over the images of the folder  

    datagen = ImageDataGenerator( rescale =1./255)
    set = datagen.flow_from_directory(path,target_size = (resol,resol),
    batch_size = counter,class_mode = 'binary', color_mode='rgb')

    return set

#-----------------------------------------------------------------------------

def store (iter) : 

    # Storing all the information in arrays for convenience 
    # returns arrays 

    X_iter , y_iter = iter.next()

    print("Shape of X_train : ", X_iter.shape)
    print("Shape of y_train : ", y_iter.shape)

    return X_iter, y_iter

#-----------------------------------------------------------------------------

def accessing_2 (path) : 

    # Retrieving the names of all files finishing with .ppm in the mentionned Directory 
    # returns a list of string 

    names = glob(path + "\\**\\*.ppm") 
    print("number of images in the folder : ", len(names))

    return names 

#-----------------------------------------------------------------------------

def store_2 (names, X_array, y_array, resol) : 

    # We'll process each image 
    # returns 2 arrays 

    for i in range(len(names)):

        # We store the category 

        # dirname(realpath()) gives the current location of the name that we process 
        # basename gives the name of the upper directory which is the category that we must predict 

        y_array.append((float(basename(dirname(realpath(names[i]))))))

        # We open the image 
        img = imread(names[i])

        # We change the color convention for cv2 
        RGB = cvtColor(img, COLOR_BGR2RGB)

        # We resize the image 
        resized=resize(RGB, (resol,resol))

        # We store it into array 
        arr = asarray(resized)

        X_array.append(arr)

    y_array = array(y_array)
    X_array =  array(X_array)

    print("Shape of X_train : ", X_array.shape)
    print("Shape of y_train : ", y_array.shape)

    return X_array, y_array