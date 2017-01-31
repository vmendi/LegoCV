import cv2
import numpy as np
from scipy.spatial import distance


def find_corners(res, img_key):
    orig = res[img_key]
    gray = np.float32(orig)
    dst = cv2.cornerHarris(gray, 3, 7, 0.04)

    # Threshold for an optimal value, it may vary depending on the image.
    gray[dst > 0.01*dst.max()] = 0

    res['corners_img'] = gray

    return res


def find_good_features_to_track(res, img_key):
    img = res[img_key].copy()

    corners = cv2.goodFeaturesToTrack(img, maxCorners=50, qualityLevel=0.02, minDistance=5,
                                      blockSize=3, useHarrisDetector=True, k=0.08)

    if corners is not None and len(corners) > 0:
        corners = np.int0(corners)
        corners = corners.squeeze()

        for i in corners:
            x,y = i.ravel()
            cv2.circle(img, (x,y), 3, 0, -1)
    else:
        corners = []

    res['corners'] = corners
    res['corners_img'] = img

    return res


def match_corners(res_a, res_b):
    width_a = res_a['corners_img'].shape[1]
    height_a = res_a['corners_img'].shape[0]

    width_b = res_b['corners_img'].shape[1]
    height_b = res_b['corners_img'].shape[0]

    # Scale invariance
    normalized_a = res_a['corners'] / [width_a - 1, height_a - 1]
    normalized_b = res_b['corners'] / [width_b - 1, height_b - 1]

    def match_corners_inner(corners_a, corners_b):
        matches_indices = []

        for index_a, corner_a in enumerate(corners_a):
            min_dist = 1_000_000
            min_index_b = -1

            for index_b, corner_b in enumerate(corners_b):
                dist = distance.euclidean(corner_a, corner_b)

                if dist < min_dist and dist < 0.10:
                    min_dist = dist
                    min_index_b = index_b

            if min_index_b != -1:
                matches_indices.append([index_a, min_index_b, min_dist])

        max_index_count = max(len(corners_a), len(corners_b))
        percentage_matched = len(matches_indices)/max_index_count

        return percentage_matched, matches_indices

    first_percentage_matched, first_matches = match_corners_inner(normalized_a, normalized_b)

    # Flip image A (ensure_max_width sets the left to right axis as the longest coordinate)
    second_percentage_matched, second_matches = match_corners_inner([1, 1] - normalized_a, normalized_b)

    if first_percentage_matched > second_percentage_matched:
        return first_percentage_matched, first_matches
    else:
        return second_percentage_matched, second_matches

# def find_corners_subpixel(img_idx, img, gray):
#     img = img.copy()
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, blockSize=10, ksize=10, k=0.04)
#     dst = cv2.dilate(dst, None)
#     ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
#     dst = np.uint8(dst)
#
#     # find centroids
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
#     # define the criteria to stop and refine the corners
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
#
#     # Now draw them
#     res = np.hstack((centroids, corners))
#     res = np.int8(res)
#     img[res[:, 1],res[:, 0]] = [0, 0, 255]
#     img[res[:, 3],res[:, 2]] = [0, 255, 0]
#
#     cv2.imwrite('out/{0}-corners.png'.format(img_idx), img)