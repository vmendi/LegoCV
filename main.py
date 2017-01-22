from os import listdir
from os.path import isfile
from time import strftime

import cv2
import numpy as np

import video
from calibration import calibrate
from find_corners import find_corners
from find_max_contour import find_max_contour, save_to_file_max_contour, clip_img, find_moments, align_and_clip
from sift import match_with_sift, find_sift


def load_and_prepare_img(src_img_filename):
    src_img = cv2.imread(src_img_filename)
    height, width, color_depth = src_img.shape
    src_img = src_img[100:height-100, 500:width-500]
    grey = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return {
        'original_img': src_img,
        'grey_img': grey,
        'filename': src_img_filename
    }


def iterate_over_images_detection(img_names):
    for img_idx, img_name in enumerate(img_names):
        res = load_and_prepare_img(img_name)
        res = find_max_contour(res)
        save_to_file_max_contour(res, img_idx)

        corners_result = find_corners(res)
        cv2.imwrite('out/{0}-corners.png'.format(img_idx), corners_result['corners'])

        sift_result = find_sift(res)
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

    training_set = [find_max_contour(res) for res in training_set]
    query_set = [find_max_contour(res) for res in query_set]

    training_set = [clip_img(res, 'original_img') for res in training_set]
    query_set = [clip_img(res, 'original_img') for res in query_set]

    match_with_sift(training_set, query_set)


def image_moments_detection(training_set_filenames, query_set_filenames):
    training_set = [load_and_prepare_img(filename) for filename in training_set_filenames]
    query_set = [load_and_prepare_img(filename) for filename in query_set_filenames]

    training_set = [find_max_contour(res) for res in training_set]
    query_set = [find_max_contour(res) for res in query_set]

    training_set = [align_and_clip(res, 'grey_img') for res in training_set]
    query_set = [align_and_clip(res, 'grey_img') for res in query_set]

    training_set = [find_moments(res, 'clipped_threshold_img', binary_image=True) for res in training_set]
    query_set = [find_moments(res, 'clipped_threshold_img', binary_image=True) for res in query_set]

    for query_idx, query in enumerate(query_set):

        print(f"Query moments:\n{query['hu_moments']}")

        filtered_training_set = filter_by_dimensions(query, training_set)
        #filtered_training_set = sort_by_area_moment(query, filtered_training_set)

        filtered_training_set = sort_by_match_shapes(query, filtered_training_set, 'clipped_threshold_img')

        cv2.imwrite(f'out/{query_idx}-query.png', query['original_img'])
        cv2.imwrite(f'out/{query_idx}-query-clipped.png', query['clipped_threshold_img'])

        if len(filtered_training_set) >= 1:
            best_match = filtered_training_set[0]
            print(f"Best match moments:\n{best_match['hu_moments']}")
            cv2.imwrite(f'out/{query_idx}-best_match.png', best_match['original_img'])
            cv2.imwrite(f'out/{query_idx}-best-match-clipped.png', best_match['clipped_threshold_img'])


def sort_by_area_moment(query, tranining_set):
    def sorter_func(a):
        return np.math.pow(query['moments']['m00'] - a['moments']['m00'], 2)

    return sorted(tranining_set, key=sorter_func)


def sort_by_match_shapes(query, training_set, img_key):
    def sorter_func(a):
        match_coeff = cv2.matchShapes(a[img_key], query[img_key], method=1, parameter=0.0)
        print(f"MatchCoeff: {match_coeff}")
        return match_coeff

    return sorted(training_set, key=sorter_func)


def filter_by_dimensions(query, training_set):
    filtered = []
    for training in training_set:
        w_ratio, h_ratio = get_dimensions_ratio(training['contour_rect'], query['contour_rect'])

        if w_ratio < 1.1 and h_ratio < 1.1:
            filtered.append(training)

    return filtered


def get_dimensions_ratio(training_rect, query_rect):
    def abs_ratio(a, b):
        return a / b if a > b else b / a

    query_width_rect = max(query_rect[1][0], query_rect[1][1])
    query_height_rect = min(query_rect[1][0], query_rect[1][1])

    training_width_rect = max(training_rect[1][0], training_rect[1][1])
    training_height_rect = min(training_rect[1][0], training_rect[1][1])

    return abs_ratio(query_width_rect, training_width_rect), abs_ratio(query_height_rect, training_height_rect)

if __name__ == '__main__':
    training_filenames = ['in/controlled_more/' + file for file in listdir('in/controlled_more') if "_00" in file]
    training_filenames = [file for file in training_filenames if isfile(file)]

    # training_filenames = [
    #                       # 'in/controlled_more/2016-12-22 23-10-06_00.png',
    #                       # 'in/controlled_more/2016-12-22 23-10-37_00.png',
    #                       # 'in/controlled_more/2016-12-22 23-08-55_00.png',
    #                       #'in/controlled_more/2016-12-22 23-10-37_00.png',
    #                       'in/controlled_more/2016-12-22 23-08-00_00.png',
    #                       ]
    query_filenames = ['in/controlled/' + file for file in listdir('in/controlled') if "_00" in file]
    query_filenames = [file for file in query_filenames if isfile(file)]
    #query_filenames = ['in/controlled/2016-12-21 13-58-28_00.png']
    #query_filenames = ['in/controlled/2016-12-21 13-55-27_00.png']


    # iterate_sift(training_filenames, query_filenames)
    # iterate_over_images_detection(training_filenames)
    # realtime_detection()
    # capture()
    # image_moments_detection(training_filenames, query_filenames)
    calibrate()

cv2.destroyAllWindows()