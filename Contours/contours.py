from cv2 import imread, COLOR_BGR2GRAY, cvtColor, GaussianBlur, Canny, findContours, contourArea, CHAIN_APPROX_SIMPLE, RETR_TREE, arcLength, approxPolyDP, isContourConvex, boundingRect, minEnclosingCircle
from imutils import grab_contours
from numpy import append, pi

 #-----------------------------------------------------------------------------

def detection (names) : 

    C = [] 

    for img in names : 

        raw_image = imread(img) #load the image

        process_image = raw_image.copy() #copy the image

        #convert the image to grayscale
        gray = cvtColor(process_image, COLOR_BGR2GRAY)

        #apply a Gaussian blur 
        bilateral_filtered_image = GaussianBlur(gray,(3,3),0)

        #apply a Canny edge detector
        edge_detected_image = Canny(bilateral_filtered_image, 90, 170)

       

        #Find the contours in the image
        contours = findContours(edge_detected_image.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)
        contours = sorted(contours, key=contourArea, reverse=True)[:8]

        C.append(contours)

    return C

 #-----------------------------------------------------------------------------

def classify(contours_img) : 
    
    for i,contour in enumerate(contours_img) : 

        length = arcLength(contour,True)
        approx = approxPolyDP(contour,0.01*length, True)

        '''
        # check if it's a closed countour 
        if not isContourConvex(approx):
            continue

        # Check if the shape defined by the contour is "regular" in terms of its edges
        area = contourArea(contour)
        perimeter = arcLength(contour, True)
        circularity = 4 * pi * area / perimeter ** 2
        if circularity < 0.8:  
            continue
        '''

        if len(approx) != 0 : 

            triangles, squares, octagons, circles = forms(approx, i, contour)

            print(triangles, squares, octagons, circles) 

            return triangles, squares, octagons, circles
        
        return [],[],[],[]

 #-----------------------------------------------------------------------------

def forms(approx, i, contour) : 

    triangles = {}
    squares = {}
    octagons = {}
    circles = {}

    #The contour is a triangle
    if len(approx) == 3:
        triangles[i] = contourArea(contour)

    #The contour is a square or a non-flat rectangle
    elif len(approx) == 4:
        (x, y, w, h) = boundingRect(approx)
        ar = w / float(h)
        if (ar >= 0.8 and ar <= 1.2):
            squares[i] = contourArea(contour)
    
    #The contour is an octagon
    elif len(approx) == 8:
        octagons[i] = contourArea(contour)

    #The contour is a circle
    elif len(approx) > 8:
        (x, y), radius = minEnclosingCircle(contour)
        area = contourArea(contour)
        if radius > 0 and area / (pi * radius**2) >= 0.8:
            circles[i] = area
    
    return triangles, squares, octagons, circles

#-----------------------------------------------------------------------------
