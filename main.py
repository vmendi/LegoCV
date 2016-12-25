from os import listdir
from os.path import isfile
from time import strftime

import cv2
import numpy as np

import video
from find_corners import find_corners, find_sift
from find_max_contour import find_max_contour


def save_to_file_max_contour(max_contour_result, img_idx):
    img_01 = max_contour_result['contours']
    img_02 = max_contour_result['original']
    img_03 = max_contour_result['threshold']
    img_04 = max_contour_result['edges']

    combined = np.zeros((img_01.shape[0] + img_03.shape[0], img_01.shape[1] + img_02.shape[1]), np.uint8)
    combined[:img_01.shape[0], :img_01.shape[1]] = img_01
    combined[:img_02.shape[0], img_01.shape[1]:img_01.shape[1] + img_02.shape[1]] = img_02
    combined[img_01.shape[0]:img_01.shape[0] + img_03.shape[0], :img_03.shape[1]] = img_03
    combined[img_01.shape[0]:img_01.shape[0] + img_04.shape[0], img_01.shape[1]:img_01.shape[1] + img_04.shape[1]] = img_04

    rect = max_contour_result['max_area_contour_rect']
    rect_text = "Bounding rect ({0:.0f},{1:.0f}), angle {2:.0f}".format(rect[1][0], rect[1][1], rect[2])

    cv2.putText(combined, rect_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite('out/{0}-combined.png'.format(img_idx), combined)

    print("Image {0} {1}".format(img_idx, rect_text))


def iterate_over_images_detection(img_names):
    for img_idx, img_name in enumerate(img_names):
        img = cv2.imread(img_name)

        height, width, color_depth = img.shape
        img = img[100:height-100, 400:width-400]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_contour_result = find_max_contour(gray)
        save_to_file_max_contour(max_contour_result, img_idx)

        corners_result = find_corners(gray)
        cv2.imwrite('out/{0}-corners.png'.format(img_idx), corners_result['corners'])

        sift_result = find_sift(gray)
        cv2.imwrite('out/{0}-sift.png'.format(img_idx), sift_result['sift'])

def realtime_detection():
    cap = video.create_capture(1)

    while True:
        flag, captured_img = cap.read()

        height, width, color_depth = captured_img.shape
        captured_img = captured_img[100:height-100, 400:width-400]

        gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
        detection_result = find_max_contour(gray)

        cv2.imshow('main', captured_img)
        cv2.imshow('threshold', cv2.resize(detection_result['threshold'], (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('edges', cv2.resize(detection_result['edges'], (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('contours', cv2.resize(detection_result['contours'], (0, 0), fx=0.5, fy=0.5))

        ch = cv2.waitKey(1)
        if ch == 27:
            break


def train_sift(training_set):
    sift_training = []
    for img_idx, img_name in enumerate(training_set):
        src_img = cv2.imread(img_name)
        height, width, color_depth = src_img.shape
        src_img = src_img[100:height-100, 500:width-500]
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        sift_result = find_sift(gray)
        sift_training.append(sift_result)

    return sift_training


def match_with_sift(training_set, query_set):

    sift_training_set = train_sift(training_set)
    sift_query_set = train_sift(query_set)

    for img_idx, sift_result in enumerate(sift_training_set):
        cv2.imwrite('out/{0}-sift-trained.png'.format(img_idx), sift_result['sift'])

    for img_idx, sift_result in enumerate(sift_query_set):
        cv2.imwrite('out/{0}-sift-queried.png'.format(img_idx), sift_result['sift'])

    for sift_query_idx, sift_query in enumerate(sift_query_set):
        bfm = cv2.BFMatcher(cv2.NORM_L2) # , crossCheck=True)

        all_trained_match_result = []
        for (sift_training_idx, sift_training) in enumerate(sift_training_set):
            matches = bfm.knnMatch(sift_training['descriptors'], sift_query['descriptors'], k=2)

            good_matches = []
            avg_distance = 0.0
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
                    avg_distance += m.distance

            if len(good_matches):
                avg_distance /= len(good_matches)
            else:
                avg_distance = float('inf')

            all_trained_match_result.append(
                {
                    'sift_training': sift_training,
                    'good_matches': good_matches,
                    'avg_distance': avg_distance
                }
            )

        if len(all_trained_match_result) > 0:
            all_trained_match_result = sorted(all_trained_match_result, key = lambda x:len(x['good_matches']), reverse=True)
            #all_trained_match_result = sorted(all_trained_match_result, key = lambda x:x['avg_distance'])
            sift_training = all_trained_match_result[0]['sift_training']

            matches_img = np.zeros((1, 1), np.uint8)
            matches_img = cv2.drawMatchesKnn(sift_query['original'], sift_query['keypoints'],
                                             sift_training['original'], sift_training['keypoints'],
                                             all_trained_match_result[0]['good_matches'],
                                             matches_img)
            cv2.imwrite("out/{0}-matched.png".format(sift_query_idx), matches_img)


def capture():
    cap_cam00 = video.create_capture(1)
    cap_cam01 = video.create_capture(2)

    while True:
        flag, captured_img_00 = cap_cam00.read()
        flag, captured_img_01 = cap_cam01.read()

        if captured_img_00 is None or captured_img_01 is None:
            continue

        cv2.imshow('main_00', captured_img_00)
        cv2.imshow('main_01', cv2.resize(captured_img_01, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.imwrite('in/{0}_00.png'.format(strftime("%Y-%m-%d %H-%M-%S")), captured_img_00)
            cv2.imwrite('in/{0}_01.png'.format(strftime("%Y-%m-%d %H-%M-%S")), captured_img_01)


if __name__ == '__main__':
    training_filenames = ['in/controlled_more/' + file for file in listdir('in/controlled_more') if "_00" in file]
    training_filenames = [file for file in training_filenames if isfile(file)]

    # training_filenames = ['in/controlled_more/2016-12-22 23-08-00_00.png',
    #                       'in/controlled_more/2016-12-22 23-08-30_00.png',
    #                       'in/controlled_more/2016-12-22 23-08-55_00.png']

    query_filenames = ['in/controlled/' + file for file in listdir('in/controlled') if "_00" in file]
    query_filenames = [file for file in query_filenames if isfile(file)]
    # query_filenames = ['in/controlled/2016-12-21 13-55-27_00.png']

    match_with_sift(training_filenames, query_filenames)
    # iterate_over_images_detection(filenames)
    # realtime_detection()
    # capture()

cv2.destroyAllWindows()