from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#-----------------------------------------------------------------------------

def construct_model (nbr_class, array) : 

    model = Sequential()

    # Tune the number of filters for the second Conv2D 
    # Choose an optimal value from 64-128
    
    model.add(Conv2D(kernel_size=(6,6),filters=112, activation='relu', input_shape=array.shape[1:]))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(kernel_size=(3,3),filters=208, activation='relu'))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(kernel_size=(2,2),filters=256, activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())

    model.add(Dense(82, activation = 'relu'))
    model.add(Dense(nbr_class, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#-----------------------------------------------------------------------------