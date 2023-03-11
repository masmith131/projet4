from numpy import zeros, arange
from matplotlib.pyplot import subplots, title, xticks, legend, show
from os import listdir
from PIL import Image


#-----------------------------------------------------------------------------

def number (nbr_class, y) :     

    # Print the number of signs of each type in the sets

    nbr = zeros(nbr_class, dtype=int)

    for i in range(nbr_class) :
        nbr[i] = int((y.copy() == i).sum())  # Number of images of class i in the set 

    print("Number of each sign in the set : ")
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

def to_jpeg (folder_dir) : 

    # Here is a code to save all ppm in jpeg in a directory (must be created)

    for image in listdir(folder_dir):
        # check if the image ends with ppm
        if (image.endswith(".ppm")):
            img = Image.open(folder_dir + '/' + image)
            img.save("challenge_1/visu" + '/' + image.replace('.ppm','.jpg'), format = 'JPEG') 