import glob
from os import listdir
from os.path import isfile, basename
from time import strftime

import cv2
import numpy as np
from skimage import measure

import video
from calibration import calibrate, load_calibration, undistort_images, undistort_image
from find_corners import find_corners
from find_max_contour import find_max_contour, clip_img, find_moments, align_and_clip, find_edges, ensure_max_width
from sift import match_with_sift, find_sift
from texture import find_lbp


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
        'filename': src_img_filename,
        'basename': basename(src_img_filename)
    }


def iterate_sift(training_set, query_set):
    training_set = [load_and_prepare_img(img_filename) for img_filename in training_set]
    query_set = [load_and_prepare_img(img_filename) for img_filename in query_set]

    training_set = [find_max_contour(res) for res in training_set]
    query_set = [find_max_contour(res) for res in query_set]

    training_set = [clip_img(res, 'original_img') for res in training_set]
    query_set = [clip_img(res, 'original_img') for res in query_set]

    match_with_sift(training_set, query_set)


def detection(training_set_filenames, query_set_filenames):
    training_set = [load_and_prepare_img(filename) for filename in training_set_filenames]
    query_set = [load_and_prepare_img(filename) for filename in query_set_filenames]

    training_set = [find_max_contour(res) for res in training_set]
    query_set = [find_max_contour(res) for res in query_set]

    training_set = [align_and_clip(res, 'grey_img') for res in training_set]
    query_set = [align_and_clip(res, 'grey_img') for res in query_set]

    training_set = [ensure_max_width(res, 'clipped_img') for res in training_set]
    query_set = [ensure_max_width(res, 'clipped_img') for res in query_set]

    training_set = [find_edges(filename, 'clipped_img') for filename in training_set]
    query_set = [find_edges(filename, 'clipped_img') for filename in query_set]

    training_set = [find_moments(res, 'edges_img', binary_image=True) for res in training_set]
    query_set = [find_moments(res, 'edges_img', binary_image=True) for res in query_set]

    training_set = [find_lbp(res, 'clipped_img', num_points=16, radius=8) for res in training_set]
    query_set = [find_lbp(res, 'clipped_img', num_points=16, radius=8) for res in query_set]

    [cv2.imwrite(f'out/{res["basename"]}-clipped.png', res['clipped_img']) for res in training_set]

    for query_idx, query in enumerate(query_set):

        # print(f"Query moments:\n{query['hu_moments']}")
        print(f"Query {query['filename']} histogram:\n{query['hist']}")
        cv2.imwrite(f'out/{query["basename"]}-query.png', query['clipped_img'])

        filtered_training_set = filter_by_dimensions(query, training_set)

        # filtered_training_set = sort_by_match_shapes(query, filtered_training_set, 'edges_img')
        # filtered_training_set = sort_by_histogram_chi_squared_distance(query, filtered_training_set)
        filtered_training_set = sort_by_mse(query, filtered_training_set)

        if len(filtered_training_set) >= 1:
            best_match = filtered_training_set[0]
            # print(f"Best match moments:\n{best_match['hu_moments']}")
            cv2.imwrite(f'out/{query["basename"]}-best-match.png', best_match['clipped_img'])

        # if len(sorted_by_match_shapes) >= 2:
        #     second_match = sorted_by_match_shapes[1]
        #     cv2.imwrite(f'out/{query_idx}-second-match.png', second_match['grey_img'])
        #     cv2.imwrite(f'out/{query_idx}-second-match-edges.png', second_match['edges_img'])


def sort_by_mse(query, training_set):
    def sorter_func(a):
        query_img = query['clipped_img']
        a_img = a['clipped_img']

        min_width = min(query_img.shape[1], a_img.shape[1])
        min_height = min(query_img.shape[0], a_img.shape[0])

        same_dimensions_a = a_img[0:min_height, 0:min_width]
        same_dimensions_query = query_img[0:min_height, 0:min_width]

        similarity = measure.compare_mse(same_dimensions_query, same_dimensions_a)

        print(f"Similarity for {a['filename']}: {similarity}")

        return similarity

    return sorted(training_set, key=sorter_func)


def sort_by_histogram_chi_squared_distance(query, training_set):
    def sorter_func(a):
        def chi2_distance(histA, histB, eps = 1e-10):
            chi_dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
            print(f"ChiDist for {a['filename']}: {chi_dist}")
            return chi_dist

        return chi2_distance(query['hist'], a['hist'])

    return sorted(training_set, key=sorter_func)


def sort_by_match_shapes(query, training_set, img_key):
    def sorter_func(a):
        match_coeff = cv2.matchShapes(a[img_key], query[img_key], method=1, parameter=0.0)
        print(f"MatchCoeff for {a['filename']}: {match_coeff}")
        return match_coeff

    return sorted(training_set, key=sorter_func)


def sort_by_area_moment(query, tranining_set):
    def sorter_func(a):
        return np.math.pow(query['moments']['m00'] - a['moments']['m00'], 2)

    return sorted(tranining_set, key=sorter_func)


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


def iterate_over_images_quick_test(img_names):
    for img_idx, img_name in enumerate(img_names):
        res = load_and_prepare_img(img_name)

        find_max_contour(res)
        align_and_clip(res, 'grey_img')

        find_corners(res, 'clipped_img')
        cv2.imwrite('out/{0}-corners.png'.format(img_idx), res['corners_img'])

        find_sift(res, 'clipped_img')
        cv2.imwrite('out/{0}-sift.png'.format(img_idx), res['sift_img'])


if __name__ == '__main__':
    training_filenames = ['in/control/' + file for file in listdir('in/control')]
    training_filenames = [file for file in training_filenames if isfile(file)]

    training_filenames = [
                          'in/control/10.bmp',
                          'in/control/13.bmp',
                          'in/control/21.bmp',
                          'in/control/22.bmp',
                          ]

    query_filenames = ['in/trial/' + file for file in listdir('in/trial')]
    query_filenames = [file for file in query_filenames if isfile(file)]
    query_filenames = [
                         'in/trial/01.bmp',
                         'in/trial/02.bmp',
                         'in/trial/05.bmp',
                         'in/trial/07.bmp',
                      ]

    # iterate_sift(training_filenames, query_filenames)
    detection(training_filenames, query_filenames)
    # calibrate()

    # camera_matrix, dist_coefs = load_calibration()
    # undistort_images(camera_matrix, dist_coefs, glob.glob('in/trial/*.bmp'))

cv2.destroyAllWindows()