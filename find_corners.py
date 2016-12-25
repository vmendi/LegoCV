import cv2
import numpy as np


def find_corners(orig_gray):
    gray = np.float32(orig_gray)
    dst = cv2.cornerHarris(gray, 3, 7, 0.04)

    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    gray[dst > 0.01*dst.max()] = 255

    return {
        'corners': gray,
        'original': orig_gray
    }


def find_corners_subpixel(img_idx, img, gray):
    img = img.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=10, k=0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int8(res)
    img[res[:, 1],res[:, 0]] = [0, 0, 255]
    img[res[:, 3],res[:, 2]] = [0, 255, 0]

    cv2.imwrite('out/{0}-corners.png'.format(img_idx), img)