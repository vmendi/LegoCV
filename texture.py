import cv2
from skimage import feature
import numpy as np
import matplotlib.pyplot as plt


def find_lbp(res, img_key, num_points, radius, eps=1e-7):
    image = res[img_key]

    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform") # method="uniform"
    as_uint8 = lbp.astype("uint8")
    hist = cv2.calcHist(images=[as_uint8], channels=[0], mask=None, histSize=[num_points+2], ranges=[0, num_points+2])

    hist = hist.flatten()
    hist_normalized = np.zeros(1)
    hist_normalized = cv2.normalize(hist, dst=hist_normalized, alpha=1, norm_type=cv2.NORM_L1)

    res['lbp'] = lbp
    res['hist'] = hist_normalized
    res['lbp_img'] = lbp * 5

    return res


def save_hist(res):
    lbp = res['lbp']
    lbp = lbp.astype("uint8")
    lbp = lbp.ravel()
    plt.hist(lbp, bins=len(res), range=[0, len(res)], normed=True)
    plt.title(f"Histogram for {res['basename']}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

