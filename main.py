from os import listdir
from os.path import isfile
from time import strftime

import cv2
import numpy as np

import video
from find_corners import find_corners
from find_max_contour import find_max_contour
from sift import match_with_sift, find_sift


def load_and_prepare_img(src_img_filename):
    src_img = cv2.imread(src_img_filename)
    height, width, color_depth = src_img.shape
    src_img = src_img[100:height-100, 500:width-500]
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return gray


def extract_max_contour(gray):
    contour_result = find_max_contour(gray)
    rect = contour_result['bounding_box']
    clipped_img = gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return clipped_img


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

    rect = max_contour_result['contour_rect']
    rect_text = "Bounding rect ({0:.0f},{1:.0f}), angle {2:.0f}".format(rect[1][0], rect[1][1], rect[2])

    cv2.putText(combined, rect_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite('out/{0}-combined.png'.format(img_idx), combined)

    print("Image {0} {1}".format(img_idx, rect_text))


def iterate_over_images_detection(img_names):
    for img_idx, img_name in enumerate(img_names):
        gray = load_and_prepare_img(img_name)
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


def iterate_sift(training_set, query_set):
    training_set = [extract_max_contour(load_and_prepare_img(img_filename)) for img_filename in training_set]
    query_set = [extract_max_contour(load_and_prepare_img(img_filename)) for img_filename in query_set]

    match_with_sift(training_set, query_set)


def image_moments_detection(training_set_filenames, query_set_filenames):
    training_set = [find_max_contour(load_and_prepare_img(filename)) for filename in training_set_filenames]
    query_set = [find_max_contour(load_and_prepare_img(filename)) for filename in query_set_filenames]

    for query_idx, query in enumerate(query_set):
        min_match_coeff = 1000000
        best_match = None
        for training in training_set:
            if are_rotated_rects_within_threshold(query['contour_rect'], training['contour_rect'], 0.20):
                match_coeff = cv2.matchShapes(training['contour'], query['contour'], method=1, parameter=0.0)
                if match_coeff < min_match_coeff:
                    min_match_coeff = match_coeff
                    best_match = training

        cv2.imwrite('out/{0}-query.png'.format(query_idx), query['original'])
        # cv2.imwrite('out/{0}-query-contours.png'.format(query_idx), query['contours'])

        if best_match:
            cv2.imwrite('out/{0}-best_match.png'.format(query_idx), best_match['original'])
            # cv2.imwrite('out/{0}-best-match-contours.png'.format(query_idx), best_match['contours'])


def are_rotated_rects_within_threshold(query_rect, training_rect, percentage_threshold):
    query_width_rect = max(query_rect[1][0], query_rect[1][1])
    query_height_rect = min(query_rect[1][0], query_rect[1][1])

    training_width_rect = max(training_rect[1][0], training_rect[1][1])
    training_height_rect = min(training_rect[1][0], training_rect[1][1])

    width_in_threshold = abs(query_width_rect - training_width_rect) < query_width_rect*percentage_threshold
    height_in_threshold = abs(query_height_rect - training_height_rect) < query_height_rect*percentage_threshold

    return width_in_threshold and height_in_threshold

if __name__ == '__main__':
    training_filenames = ['in/controlled_more/' + file for file in listdir('in/controlled_more') if "_00" in file]
    training_filenames = [file for file in training_filenames if isfile(file)]

    # training_filenames = ['in/controlled_more/2016-12-22 23-08-00_00.png',
    #                       'in/controlled_more/2016-12-22 23-08-30_00.png',
    #                       'in/controlled_more/2016-12-22 23-08-55_00.png',
    #                       'in/controlled_more/2016-12-22 23-10-06_00.png',
    #                       'in/controlled_more/2016-12-22 23-12-10_00.png']

    query_filenames = ['in/controlled/' + file for file in listdir('in/controlled') if "_00" in file]
    query_filenames = [file for file in query_filenames if isfile(file)]
    # query_filenames = ['in/controlled/2016-12-21 13-55-27_00.png']
    # query_filenames = ['in/uncontrolled/2016-12-21 12-51-37_00.png']

    # iterate_sift(training_filenames, query_filenames)
    # iterate_over_images_detection(training_filenames)
    # realtime_detection()
    # capture()
    image_moments_detection(training_filenames, query_filenames)

cv2.destroyAllWindows()