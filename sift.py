import cv2
import numpy as np
from os.path import basename

from find_max_contour import find_max_contour


def find_sift(orig_gray):
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10,sigma=1.6)
    sift = cv2.xfeatures2d.SIFT_create()

    # http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    # http://docs.opencv.org/trunk/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    (keypoints, descs) = sift.detectAndCompute(orig_gray, None)

    img = orig_gray.copy()
    img = cv2.drawKeypoints(orig_gray, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return {
        'keypoints': keypoints,
        'descriptors': descs,
        'sift': img,
        'original': orig_gray
    }


def load_and_prepare_img(src_img_filename):
    src_img = cv2.imread(src_img_filename)
    height, width, color_depth = src_img.shape
    src_img = src_img[100:height-100, 500:width-500]
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return gray


def extract_max_contour(src_img_filename):
    gray = load_and_prepare_img(src_img_filename)
    contour_result = find_max_contour(gray)
    rect = contour_result['max_area_bounding_box']
    clipped_img = gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return clipped_img


def match_with_sift(training_set, query_set):

    training_set = [extract_max_contour(img_filename) for img_filename in training_set]
    query_set = [extract_max_contour(img_filename) for img_filename in query_set]

    sift_training_set = [find_sift(img) for img in training_set]
    sift_query_set = [find_sift(img) for img in query_set]

    for img_idx, sift_result in enumerate(sift_training_set):
        cv2.imwrite('out/{0}-sift-trained.png'.format(img_idx), sift_result['sift'])

    for img_idx, sift_result in enumerate(sift_query_set):
        cv2.imwrite('out/{0}-sift-queried.png'.format(img_idx), sift_result['sift'])

    for sift_query_idx, sift_query in enumerate(sift_query_set):
        bfm = cv2.BFMatcher(cv2.NORM_L2) # , crossCheck=True)

        match_results = []
        for (sift_training_idx, sift_training) in enumerate(sift_training_set):
            matches = bfm.knnMatch(sift_query['descriptors'], sift_training['descriptors'], k=2)

            good_matches = []
            query_keypoints = []
            training_keypoints = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
                    query_keypoints.append(sift_query['keypoints'][m.queryIdx].pt)
                    training_keypoints.append(sift_training['keypoints'][m.trainIdx].pt)

            training_keypoints = np.array(training_keypoints)
            query_keypoints = np.array(query_keypoints)

            mask = None
            if len(training_keypoints) > 0 and len(query_keypoints) > 0:
                _, mask = cv2.findHomography(srcPoints=training_keypoints, dstPoints=query_keypoints,
                                             method=cv2.RANSAC, ransacReprojThreshold=5)

            match_results.append(
                {
                    'sift_training': sift_training,
                    'good_matches': good_matches,
                    'mask': mask
                }
            )

        best_match = max(match_results, key=lambda match: np.count_nonzero(match['mask']))

        matches_img = np.zeros((1, 1), np.uint8)
        matches_img = cv2.drawMatchesKnn(sift_query['original'],
                                         sift_query['keypoints'],
                                         best_match['sift_training']['original'],
                                         best_match['sift_training']['keypoints'],
                                         best_match['good_matches'],
                                         matches_img)
        cv2.imwrite("out/{0}-matched.png".format(sift_query_idx), matches_img)
