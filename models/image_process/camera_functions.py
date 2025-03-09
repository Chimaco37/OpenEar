import cv2 as cv
import glob
import numpy as np

def undistort_and_resize_image(fname, mtx, dist, width_size, height_size):
    img = cv.imread(fname)

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    dst = cv.resize(dst, (width_size, height_size))
    output_name = fname.replace('frame', 'undistort')
    cv.imwrite(output_name, dst)

def resize_image(fname, width_size, height_size):
    img = cv.imread(fname)

    # resize the image
    img = cv.resize(img, (width_size, height_size))
    output_name = fname.replace('_without_resize', '')
    cv.imwrite(output_name, img)
