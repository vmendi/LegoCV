from os import listdir
from os.path import isfile
from time import strftime

import cv2
import numpy as np

import video
from find_corners import find_corners
from find_max_contour import find_max_contour, save_to_file_max_contour, clip_img, find_moments
from sift import match_with_sift, find_sift


def load_and_prepare_img(src_img_filename):
    src_img = cv2.imread(src_img_filename)
    height, width, color_depth = src_img.shape
    src_img = src_img[100:height-100, 500:width-500]
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return gray


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
        cv2.imshow('threshold', cv2.resize(detection_result['threshold_img'], (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('edges', cv2.resize(detection_result['edges_img'], (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('contours', cv2.resize(detection_result['contours_img'], (0, 0), fx=0.5, fy=0.5))

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
    training_set = [load_and_prepare_img(img_filename) for img_filename in training_set]
    query_set = [load_and_prepare_img(img_filename) for img_filename in query_set]

    training_set = [find_max_contour(img) for img in training_set]
    query_set = [find_max_contour(img) for img in query_set]

    training_set = [clip_img(res, 'original_img') for res in training_set]
    query_set = [clip_img(res, 'original_img') for res in query_set]

    match_with_sift(training_set, query_set)


def image_moments_detection(training_set_filenames, query_set_filenames):
    training_set = [load_and_prepare_img(filename) for filename in training_set_filenames]
    query_set = [load_and_prepare_img(filename) for filename in query_set_filenames]

    training_set = [find_max_contour(img) for img in training_set]
    query_set = [find_max_contour(img) for img in query_set]

    training_set = [clip_img(res, 'threshold_img') for res in training_set]
    query_set = [clip_img(res, 'threshold_img') for res in query_set]

    training_set = [find_moments(res, 'max_contour', binary_image=False) for res in training_set]
    query_set = [find_moments(res, 'max_contour', binary_image=False) for res in query_set]

    for query_idx, query in enumerate(query_set):
        best_match = None
        best_coeff = 1000000

        query_moments = query['hu_moments']
        print(f"Query HuMoments: {query_moments}")

        for training in training_set:
            #match_coeff = cv2.matchShapes(training['clipped_img'], query['clipped_img'], method=1, parameter=0.0)
            match_coeff = cv2.matchShapes(training['max_contour'], query['max_contour'], method=1, parameter=0.0)
            # w_ratio, h_ratio = get_dimensions_ratio(training['contour_rect'], query['countour_rect'])
            training_moments = training['hu_moments']
            print(f"Training HuMoments: {training_moments}")
            print(f"MatchCoeff: {match_coeff}")

            if match_coeff < best_coeff:
                best_coeff = match_coeff
                best_match = training

        cv2.imwrite(f'out/{query_idx}-query.png', query['original_img'])
        cv2.imwrite(f'out/{query_idx}-query-clipped.png', query['clipped_img'])

        if best_match:
            cv2.imwrite(f'out/{query_idx}-best_match.png', best_match['original_img'])
            cv2.imwrite(f'out/{query_idx}-best-match-clipped.png', best_match['clipped_img'])


def get_dimensions_ratio(training_rect, query_rect):
    query_width_rect = max(query_rect[1][0], query_rect[1][1])
    query_height_rect = min(query_rect[1][0], query_rect[1][1])

    training_width_rect = max(training_rect[1][0], training_rect[1][1])
    training_height_rect = min(training_rect[1][0], training_rect[1][1])

    return query_width_rect/training_width_rect, query_height_rect/training_height_rect

if __name__ == '__main__':
    training_filenames = ['in/controlled_more/' + file for file in listdir('in/controlled_more') if "_00" in file]
    training_filenames = [file for file in training_filenames if isfile(file)]

    training_filenames = [
                          'in/controlled_more/2016-12-22 23-10-06_00.png',
                          'in/controlled_more/2016-12-22 23-09-55_00.png'
                          # 'in/controlled_more/2016-12-22 23-08-00_00.png',
                          # 'in/controlled_more/2016-12-22 23-08-30_00.png',
                          # 'in/controlled_more/2016-12-22 23-08-55_00.png',
                          # 'in/controlled_more/2016-12-22 23-12-10_00.png
                          ]

    query_filenames = ['in/controlled/' + file for file in listdir('in/controlled') if "_00" in file]
    query_filenames = [file for file in query_filenames if isfile(file)]
    # query_filenames = ['in/controlled/2016-12-21 13-55-27_00.png']
    query_filenames = ['in/uncontrolled/2016-12-21 12-51-37_00.png']

    # iterate_sift(training_filenames, query_filenames)
    # iterate_over_images_detection(training_filenames)
    # realtime_detection()
    # capture()
    image_moments_detection(training_filenames, query_filenames)

cv2.destroyAllWindows()