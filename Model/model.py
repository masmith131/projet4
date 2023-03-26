from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#-----------------------------------------------------------------------------

def construct_model (nbr_class, array) : 

    model = Sequential()

    model.add(Conv2D(filters=68, kernel_size=(5,5), activation='relu', input_shape=array.shape[1:]))
    model.add(Conv2D(filters=68, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(4,4)))
    model.add(Dropout(rate=0.35))

    model.add(Conv2D(filters=136, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=136, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Dropout(rate=0.4))

    model.add(Flatten())
    model.add(Dense(122, activation='relu'))

    model.add(Dense(nbr_class, activation='softmax'))

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#-----------------------------------------------------------------------------