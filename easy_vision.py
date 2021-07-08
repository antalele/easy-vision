
"""
Created on 10/06/2021

@author: Taggio Emanuele
"""
from __future__ import print_function
import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import imutils

# visualizza (stampa a schermo) l'immagine passata come parametro
# non restituisce niente
def dev_display(img):
    # false per "interattivo" (zoom ecc...)
    if(1):
        # creo una finestra
        cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
        # imposto cosa deve essere visualizzato
        cv2.imshow("Display window", img)
        # per aggiorare la finestra devo fare cv.wait(0)
        cv2.waitKey(0)
    else:
        # show the image with the drawn contours
        plt.imshow(img)
        plt.show()



# visualizza (stampa a schermo) l'immagine passata come parametro1
# e sovreppone in semi trasparenza la regione passata come parametro2
# non restituisce niente
def dev_display_region(image, region):
    alpha = 0.5
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = region.copy()
    output = image.copy()
    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    dev_display(output)
    return output

# compute the difference between two regions
def difference(region1, region2):
    diff = np.subtract(region1, region2)
    RegionDifference = diff.copy()
    RegionDifference[diff < 0] = 0

    # uint_img = np.array(RegionDifference*255).astype('uint8')
    # img = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

    img = np.array(RegionDifference * 255, dtype=np.uint8)
    # questa riga qui sotto invece sembra disegnare i contours della regione interessata
    # però dà il risultato in un canale al contrario dell'altra che lo dai in rgb
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    dev_display(img)
    return img

# create a binary image from img where img's values are between min and max
def threshold(img, min, max):
    x, region_min = cv2.threshold(img, min, 1, cv2.THRESH_BINARY)
    dev_display(region_min)
    x, region_max = cv2.threshold(img, max, 1, cv2.THRESH_BINARY)
    dev_display(region_max)
    region = difference(region_min, region_max)
    return region

# compute the opening of a region binary image: an erosion succeded by a dilation
# it uses a rectangular mask of dimension [width, height]
def opening_rectangle(region, width, height):
    # construct a rectangular kernel from the current size and then
    # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [width, height])
    opening = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
    return opening

# compute the opening of a region binary image: an erosion succeded by a dilation
# it uses a circular mask of radius [radius]
def opening_circle(region, radius):
    # construct a rectangular kernel from the current size and then
    # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [radius,radius])
    opening = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
    return opening

# compute the closing of a region binary image: a dilation succeded by a erosion
# it uses a rectangular mask of dimension [width, height]
def closing_rectangle(region, width, height):
    # construct a rectangular kernel from the current size and then
    # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [width, height])
    opening = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
    return opening

# compute the opening of a region binary image: a dilation succeded by a erosion
# it uses a circular mask of radius [radius]
def closing_circle(region, radius):
    # construct a rectangular kernel from the current size and then
    # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [radius,radius])
    opening = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
    return opening