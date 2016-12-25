import cv2
import numpy as np

def find_sift(orig_gray):
    # # http://docs.opencv.org/trunk/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=20, nOctaveLayers=5, contrastThreshold=0.01, edgeThreshold=20,sigma=2)
    sift = cv2.xfeatures2d.SIFT_create()

    # http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    (keypoints, descs) = sift.detectAndCompute(orig_gray, None)

    img = orig_gray.copy()
    img = cv2.drawKeypoints(orig_gray, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return {
        'keypoints': keypoints,
        'descriptors': descs,
        'sift': img,
        'original': orig_gray
    }


def match_with_sift(training_set, query_set):
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
                                         outImg=matches_img,
                                         matchColor=(255, 0, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("out/{0}-matched.png".format(sift_query_idx), matches_img)
