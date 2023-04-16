from numpy import zeros, where
from numpy.random import normal 
from tensorflow import gather
from random import randint
from keras.layers import RandomRotation, RandomZoom, RandomCrop, RandomBrightness, RandomContrast
from tensorflow.keras import Sequential 

#-----------------------------------------------------------------------------

def augmentation (dic) : 

    data_augmentation = Sequential(
        [
            RandomRotation((-0.05, 0.05)),
            RandomZoom(height_factor=(-0.3, -0.2),width_factor=(-0.3, -0.2)),
            RandomCrop(height=dic['resol'][0],width=dic['resol'][1]),
        ]
    )

    # This fct processes augmentation of the training set 

    aug = zeros(dic['nbr_class'])

    counter = 0 # This represents the number of elements we need to add 

    for i in range (dic['nbr_class']) : 

        if dic['initial_nbr_train'][i] >= dic['lim'] : continue # There's enough images 

        # There's not enough images so we must add some 
        # If there's already a lot of images for this category, 
        # we don't need too add many of them 

        aug[i] = int( (3/4) * (dic['lim'] - dic['initial_nbr_train'][i]) )
        counter += aug[i]

    X_train_new = zeros((int(len(dic['X_train'])) + counter, int(dic['resol'][0][0]), int(dic['resol'][1][0]), 3))
    y_train_new = zeros(len(dic['y_train'] + counter))
    X_train_new[:len(dic['X_train'])] = dic['X_train']
    y_train_new[:len(dic['y_train'])] = dic['y_train']

    index = len(dic['X_train'])

    for i in range (dic['nbr_class']) : 

        indices = where(dic['y_train'] == i)

        for j in range (int(dic['prop']*aug[i])) : 

            idx = randint(0, len(indices[0])-1)
            augmented_image = data_augmentation(dic['X_train'][indices[0][idx]])
            X_train_new[index] = augmented_image
            y_train_new[index] = i
            index += 1

        for j in range (int(dic['prop_noise']*aug[i])) : 

            idx = randint(0, len(indices[0])-1)
            augmented_image = dic['noise'](dic['X_train'][indices[0][idx]], dic['noise_mean'], dic['noise_var'])
            X_train_new[index] = data_augmentation(augmented_image)
            y_train_new[index] = i
            index += 1

        for j in range (int(dic['prop_blur']*aug[i])) : 

            idx = randint(0, len(indices[0])-1)
            augmented_image = dic['blur'](dic['X_train'][indices[0][idx]], dic['resol'], dic['blur_sigma'])
            X_train_new[index] = data_augmentation(augmented_image)
            y_train_new[index] = i
            index += 1

    print("Shape of the augmented training set with second method : ", X_train_new.shape)
    print("Shape of the augmented training target with second method : ", y_train_new.shape)

    return X_train_new, y_train_new