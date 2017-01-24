import glob
from os import listdir
from os.path import isfile
from time import strftime

import cv2
import numpy as np

import video
from calibration import calibrate, load_calibration, undistort_images, undistort_image
from find_corners import find_corners
from find_max_contour import find_max_contour, clip_img, find_moments, align_and_clip, find_edges
from sift import match_with_sift, find_sift


def load_and_prepare_img(src_img_filename):
    src_img = cv2.imread(src_img_filename)

    camera_matrix, dist_coefs = load_calibration()
    undistorted = undistort_image(camera_matrix, dist_coefs, src_img)

    height, width, color_depth = undistorted.shape
    clipped = undistorted[10:height-10, 100:width-100]

    grey = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)

    return {
        'original_img': src_img,
        'grey_img': grey,
        'filename': src_img_filename
    }


def iterate_over_images_detection(img_names):
    for img_idx, img_name in enumerate(img_names):
        res = load_and_prepare_img(img_name)
        res = find_max_contour(res)
        save_to_file(res, img_idx)

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
        cv2.imshow('edges', cv2.resize(detection_result['edges_threshold_img'], (0, 0), fx=0.5, fy=0.5))
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

    training_set = [find_edges(filename, 'clipped_img') for filename in training_set]
    query_set = [find_edges(filename, 'clipped_img') for filename in query_set]

    training_set = [find_moments(res, 'edges_img', binary_image=True) for res in training_set]
    query_set = [find_moments(res, 'edges_img', binary_image=True) for res in query_set]

    for query_idx, query in enumerate(query_set):

        print(f"Query moments:\n{query['hu_moments']}")

        filtered_training_set = filter_by_dimensions(query, training_set)
        #filtered_training_set = sort_by_area_moment(query, filtered_training_set)

        sorted_by_match_shapes = sort_by_match_shapes(query, filtered_training_set, 'edges_img')

        cv2.imwrite(f'out/{query_idx}-query.png', query['grey_img'])
        cv2.imwrite(f'out/{query_idx}-query-edges.png', query['edges_img'])

        if len(sorted_by_match_shapes) >= 1:
            best_match = sorted_by_match_shapes[0]
            print(f"Best match moments:\n{best_match['hu_moments']}")
            cv2.imwrite(f'out/{query_idx}-best-match.png', best_match['grey_img'])
            cv2.imwrite(f'out/{query_idx}-best-match-edges.png', best_match['edges_img'])

        if len(sorted_by_match_shapes) >= 2:
            second_match = sorted_by_match_shapes[1]
            cv2.imwrite(f'out/{query_idx}-second-match.png', second_match['grey_img'])
            cv2.imwrite(f'out/{query_idx}-second-match-edges.png', second_match['edges_img'])


def sort_by_area_moment(query, tranining_set):
    def sorter_func(a):
        return np.math.pow(query['moments']['m00'] - a['moments']['m00'], 2)

    return sorted(tranining_set, key=sorter_func)


def sort_by_match_shapes(query, training_set, img_key):
    def sorter_func(a):
        match_coeff = cv2.matchShapes(a[img_key], query[img_key], method=1, parameter=0.0)
        print(f"MatchCoeff for {a['filename']}: {match_coeff}")
        return match_coeff

    return sorted(training_set, key=sorter_func)


def filter_by_dimensions(query, training_set):
    filtered = []
    for training in training_set:
        w_ratio, h_ratio = get_dimensions_ratio(training['contour_rect'], query['contour_rect'])

        if w_ratio < 1.05 and h_ratio < 1.05:
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


def save_to_file(res, img_idx):
    img_01 = res['grey_img']
    img_02 = res['threshold_img']
    img_03 = res['edges_threshold_img']
    img_04 = res['contours_img']

    combined = np.zeros((img_01.shape[0] + img_03.shape[0], img_01.shape[1] + img_02.shape[1]), np.uint8)
    combined[:img_01.shape[0], :img_01.shape[1]] = img_01
    combined[:img_02.shape[0], img_01.shape[1]:img_01.shape[1] + img_02.shape[1]] = img_02
    combined[img_01.shape[0]:img_01.shape[0] + img_03.shape[0], :img_03.shape[1]] = img_03
    combined[img_01.shape[0]:img_01.shape[0] + img_04.shape[0], img_01.shape[1]:img_01.shape[1] + img_04.shape[1]] = img_04

    rect = res['contour_rect']
    rect_text = "Bounding rect ({0:.0f},{1:.0f}), angle {2:.0f}".format(rect[1][0], rect[1][1], rect[2])

    cv2.putText(combined, rect_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite('out/{0}-combined.png'.format(img_idx), combined)

    print("Image {0} {1}".format(img_idx, rect_text))


def edges_quick_test(training_set_filenames):
    training_set = [load_and_prepare_img(filename) for filename in training_set_filenames]
    training_set = [find_edges(res, 'grey_img') for res in training_set]

    [cv2.imwrite('out/{0}-edges.png'.format(img_idx), img['edges_img']) for img_idx, img in enumerate(training_set)]


if __name__ == '__main__':
    training_filenames = ['in/control/' + file for file in listdir('in/control')]
    training_filenames = [file for file in training_filenames if isfile(file)]

    # training_filenames = [
    #                       'in/control/00.bmp',
    #                       ]


    query_filenames = ['in/trial/' + file for file in listdir('in/trial')]
    query_filenames = [file for file in query_filenames if isfile(file)]
    # query_filenames = [
    #                     'in/trial/05.bmp'
    #                   ]


    # iterate_sift(training_filenames, query_filenames)
    # iterate_over_images_detection(training_filenames)
    # realtime_detection()
    # capture()
    # image_moments_detection(training_filenames, query_filenames)
    # calibrate()

    # edges_quick_test(training_filenames)

    # camera_matrix, dist_coefs = load_calibration()
    # undistort_images(camera_matrix, dist_coefs, glob.glob('in/trial/*.bmp'))

cv2.destroyAllWindows()