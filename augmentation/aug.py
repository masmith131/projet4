from numpy import zeros, where, abs
from tensorflow import gather
from random import randint

#-----------------------------------------------------------------------------

def first (nbr_class, initial_nbr_train, initial_nbr_test, X_train, y_train, resol) : 

    # First method of augmentation 
    # If the training set has less than 3/4 of elements compared to the test set, we fill it with augmented images 

    memory = zeros((nbr_class, 2), dtype=int)
    counter_train = 0

    

    for i in range(nbr_class) :
        
        if initial_nbr_train[i] < 3*initial_nbr_test[i]/4 :  

            if len(where(y_train == i)[0]) != 0 : # This would mean that we wouldn't see the sign in the training set 

                memory[i][0] = abs(3*initial_nbr_test[i]/4 - initial_nbr_train[i])
                counter_train += int(abs(3*initial_nbr_test[i]/4 - initial_nbr_train[i]))

   

    # We create new sets that we'll fill with the data of the initial sets + the augmented data 

    X_train_first = zeros((len(X_train) + counter_train, resol[0], resol[1], 3))
    y_train_first = zeros(len(y_train) + counter_train)

    X_train_first[:len(X_train)] = X_train.copy()
    y_train_first[:len(y_train)] = y_train.copy()

    

    # We'll start adding values at this index

    index_train = len(X_train)

    nbr_train_first = initial_nbr_train.copy()

    

    for i in range(nbr_class) :

        if memory[i][0] > 0 : # We'll add it to the new set 
            
            indices = where(y_train == i) 
            augmented_image = gather(X_train.copy(), indices=indices[0])

            for j in range(memory[i][0]) :

                nbr_train_first[i] += 1 

                idx = randint(0, len(indices[0])-1)

                X_train_first[index_train] = augmented_image[idx]
                y_train_first[index_train] = i
                index_train += 1
                
   

    print("Shape of the augmented training set with first method : ", X_train_first.shape)
    print("Shape of the augmented training target with first method : ", y_train_first.shape)

    return X_train_first, y_train_first

#-----------------------------------------------------------------------------

def second(nbr_class, initial_nbr_train,X_train, y_train, resol, num) : 

    # Second method of data augmentation 
    # All classes are represented with the same number of sign, the num parameter
    memory = zeros(nbr_class, dtype=int)
    lim = num
    counter = 0

    

    for i in range(nbr_class) :

        memory[i] = lim - initial_nbr_train[i]
        counter += lim - initial_nbr_train[i]

   

    # We create new sets that we'll fill with the data of the initial sets + the augmented data

    X_train_second = zeros((len(X_train) + counter, resol[0], resol[1], 3))
    y_train_second = zeros(len(y_train) + counter)

    X_train_second[:len(X_train)] = X_train.copy()
    y_train_second[:len(y_train)] = y_train.copy()

   

    # We'll start adding values at this index

    index = len(X_train)

    nbr_train_second = initial_nbr_train.copy() 

    

    for i in range(nbr_class) :

        indices = where(y_train == i)
        augmented_image = gather(X_train.copy(), indices=indices[0])

        for j in range(memory[i]) :

            idx = randint(0, len(indices[0])-1)

            nbr_train_second[i] += 1 

            X_train_second[index] = augmented_image[idx]
            y_train_second[index] = i
            index += 1

    

    print("Shape of the augmented training set with second method : ", X_train_second.shape)
    print("Shape of the augmented training target with second method : ", y_train_second.shape)

    return X_train_second, y_train_second

#-----------------------------------------------------------------------------