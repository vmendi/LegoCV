import cv2
from skimage import feature
import numpy as np


def find_lbp(res, img_key, num_points, radius, eps=1e-7):
    image = res[img_key]

    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform") # method="uniform"
    # (hist, _) = np.histogram(lbp.ravel(),
    #                          bins=np.arange(0, num_points + 3),
    #                          range=(0, num_points + 2))
    as_uint8 = lbp.astype("uint8")
    hist = cv2.calcHist(images=[as_uint8], channels=[0], mask=None, histSize=[num_points+3], ranges=[0, num_points + 2])

    # normalize the histogram
    # hist = hist.astype("float")
    # hist /= (hist.sum() + eps)

    hist = hist.ravel()

    res['hist'] = hist
    res['lbp_img'] = lbp * 5

    return res
