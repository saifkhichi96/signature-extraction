import cv2
import numpy as np


def remove_xlines(im):
    lines = []

    h, w = im.shape
    columns = range(w)
    rows = range(h)

    for r in rows:  # For each horizontal scan line
        mean = np.mean(im[r, columns])  # calculate pixel density (i.e.
        pxdt = mean / 255.0  # percentage of white pixels)

        if pxdt > 0.25:  # For >25% white pixels, we make the
            lines.append(r)  # whole scan line (FIXME: Should be
            im[r, columns] = 0  # the actual line only) black.

    return im, lines


def remove_ylines(im):
    lines = []

    h, w = im.shape
    columns = range(w)
    rows = range(h)

    for c in columns:  # For each vertical scan line
        mean = np.mean(im[rows, c])  # calculate pixel density (i.e.
        pxdt = mean / 255.0  # percentage of white pixels)

        if pxdt > 0.25:  # For >25% white pixels, we make the
            lines.append(c)  # whole scan line (FIXME: Should be
            im[rows, c] = 0  # the actual line only) black.

    return lines


def remove_lines(image, kernel=(25, 25)):
    # Make a copy (original image will be needed later)
    copy = np.copy(image)

    # Remove all lines (horizontal and vertical)
    x_lines = remove_xlines(copy)
    y_lines = remove_ylines(copy)

    # Remove noise (removes any parts of lines not removed)
    filter = cv2.GaussianBlur(copy, kernel, 0)
    ret3, copy = cv2.threshold(filter, 0, 255, \
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Guassian filtering for noise removal thickens all strokes
    # and filling can sometimes color pixels which were unfilled
    # in original image. These side effects are reversed by
    # taking an intersection of the processed image with the
    # original image
    return cv2.bitwise_and(copy, image)


def threshold(im):
    blur = cv2.GaussianBlur(im, (25, 25), 0)
    im_bin_1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,121,7)
    im_bin_2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,2)
    return np.bitwise_and(im_bin_1, im_bin_2)
