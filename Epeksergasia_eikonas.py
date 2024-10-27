
import cv2
import numpy as np
from skimage.morphology import skeletonize

def extract_outline(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    eroded = cv2.erode(image, kernel, iterations=1)
    outline = dilated - eroded
    return outline

def thinning(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_not(binary_image)
    skeleton = skeletonize(binary_image // 255)
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def sobel_edges(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel

def laplacian_edges(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def canny_edges(image):
    canny = cv2.Canny(image, 100, 200)
    return canny

# Διαβάζουμε την εικόνα σε grayscale
image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)  

# Εξαγωγή περιγράμματος
outline = extract_outline(image)
cv2.imwrite('outline.jpg', outline)
cv2.imshow('Outline', outline)

# Λέπτυνση
skeleton = thinning(image)
cv2.imwrite('skeleton.jpg', skeleton)
cv2.imshow('Skeleton', skeleton)

# Ανίχνευση ακμών με φίλτρα Sobel
sobel = sobel_edges(image)
cv2.imwrite('sobel.jpg', sobel)
cv2.imshow('Sobel', sobel)

# Ανίχνευση ακμών με Λαπλασιανή
laplacian = laplacian_edges(image)
cv2.imwrite('laplacian.jpg', laplacian)
cv2.imshow('Laplacian', laplacian)

# Ανίχνευση ακμών με ανιχνευτή Canny
canny = canny_edges(image)
cv2.imwrite('canny.jpg', canny)
cv2.imshow('Canny', canny)

# Εμφάνιση των εικόνων μέχρι να πατηθεί ένα πλήκτρο
cv2.waitKey(0)
cv2.destroyAllWindows()
