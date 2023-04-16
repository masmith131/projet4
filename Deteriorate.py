from numpy import zeros
from numpy.random import normal
from cv2 import GaussianBlur

def noise (img, mean, var) : 

    # This fct applies random (normal) noise on an image 

    row,col,ch= img.shape

    sigma = var**0.5

    gauss = normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noise = img + gauss

    return noise 

#-----------------------------------------------------------------------------

def blur (img, resol, sigma) : 

    # This fct applies blur on an image by applying GaussianBlur 

    blur = GaussianBlur(img,(resol+1,resol+1), sigma)

    return blur 