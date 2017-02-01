from os import listdir
from os.path import isfile, basename

import cv2
import numpy as np
import skimage
from skimage import measure

from calibration import load_calibration, undistort_image
from find_corners import find_good_features_to_track, match_corners
from find_max_contour import find_max_contour, clip_img, find_moments, align_and_clip, ensure_max_width, \
    find_edges_better, supress_background_noise
from sift import match_with_sift
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

    training_set = [supress_background_noise(res, 'grey_img') for res in training_set]
    query_set = [supress_background_noise(res, 'grey_img') for res in query_set]

    training_set = [find_max_contour(res) for res in training_set]
    query_set = [find_max_contour(res) for res in query_set]

    training_set = [align_and_clip(res, 'grey_img') for res in training_set]
    query_set = [align_and_clip(res, 'grey_img') for res in query_set]

    training_set = [ensure_max_width(res, 'clipped_img') for res in training_set]
    query_set = [ensure_max_width(res, 'clipped_img') for res in query_set]

    training_set = [find_edges_better(res, 'clipped_img') for res in training_set]
    query_set = [find_edges_better(res, 'clipped_img') for res in query_set]

    training_set = [find_moments(res, 'edges_img', binary_image=True) for res in training_set]
    query_set = [find_moments(res, 'edges_img', binary_image=True) for res in query_set]

    query_set = [find_lbp(res, 'clipped_img', num_points=16, radius=2) for res in query_set]
    training_set = [find_lbp(res, 'clipped_img', num_points=16, radius=2) for res in training_set]

    training_set = [find_good_features_to_track(res, 'clipped_img') for res in training_set]
    query_set = [find_good_features_to_track(res, 'clipped_img') for res in query_set]

    #[cv2.imwrite(f'out/{res["basename"]}-clipped.png', res['clipped_img']) for res in training_set]

    for query_idx, query in enumerate(query_set):

        print(f"Query {query['filename']}")
        # print(f"Query moments:\n{query['hu_moments']}")

        cv2.imwrite(f'out/{query["basename"]}-query.png', query['clipped_img'])
        cv2.imwrite(f'out/{query["basename"]}-lbp.png', query['lbp_img'])
        cv2.imwrite(f'out/{query["basename"]}-edges.png', query['edges_img'])
        cv2.imwrite(f'out/{query["basename"]}-corners.png', query['corners_img'])

        scores = {}

        filtered_training_set = filter_by_dimensions(query, training_set)

        #[cv2.imwrite(f'out/{res["basename"]}-lbp.png', res['lbp_img']) for res in filtered_training_set]

        if len(filtered_training_set) > 1:
            calc_corners_score(query, filtered_training_set, scores)
            filtered_training_set = filter_by_proximity_to_score(filtered_training_set, scores, 'corners_scores',
                                                                 find_top_score(scores, 'corners_scores'), 0.3)

            calc_shapes_score(query, filtered_training_set, scores)
            filtered_training_set = filter_by_score(filtered_training_set, scores, 'shapes_scores', None, 0.2)
            calc_histogram_score(query, filtered_training_set, scores)
            calc_correlation_score(query, filtered_training_set, scores)
            calc_aggregated_score(scores)

            filtered_training_set = sort_by_score(filtered_training_set, scores, 'sum_shapes_hist')

        if len(filtered_training_set) >= 1:
            best_match = filtered_training_set[0]
            print(f"Best match for {query['filename']}: {best_match['filename']}")
            cv2.imwrite(f'out/{query["basename"]}-best-match.png', best_match['clipped_img'])
            cv2.imwrite(f'out/{query["basename"]}-best-match-lbp.png', best_match['lbp_img'])
            cv2.imwrite(f'out/{query["basename"]}-best-match-edges.png', best_match['edges_img'])
            cv2.imwrite(f'out/{query["basename"]}-best-match-corners.png', best_match['corners_img'])


def sort_by_score(training_set, scores, what_to_sort, reverse=True):
    def sorter_func(a):
        return scores[what_to_sort][a['filename']]

    return sorted(training_set, key=sorter_func, reverse=reverse)


def find_top_score(scores, what_to_filter_key):
    max_score = -1
    for key, value in scores[what_to_filter_key].items():
        if value > max_score:
            max_score = value
    return max_score


def find_bottom_score(scores, what_to_filter_key):
    min_score = 100000
    for key, value in scores[what_to_filter_key].items():
        if value < min_score:
            min_score = value
    return min_score


def filter_by_proximity_to_score(training_set, scores, what_to_filter_key, top_score, percent_proximity):
    def percent_distance(a):
        if top_score == 0:
            percent = 0
        else:
            percent = a / top_score

        return abs(1 - percent) < percent_proximity

    return [close_enough for close_enough in training_set if percent_distance(scores[what_to_filter_key][close_enough['filename']])]


def filter_by_score(training_set, scores, what_to_filter_key, min_value, max_value):
    return [blah for blah in training_set
            if (min_value is None or scores[what_to_filter_key][blah['filename']] > min_value)
            and (max_value is None or scores[what_to_filter_key][blah['filename']] < max_value)]


def calc_aggregated_score(scores):
    scores['sum_shapes_hist'] = {}

    for hist_key, hist_value in scores['hist_scores'].items():
        correlation_score = scores['correlation_scores'][hist_key]
        shape_value = (1 - 5*scores['shapes_scores'][hist_key])

        if shape_value < 0:
            raise Exception("shape score has to be filtered to remove < 0.1 values")

        scores['sum_shapes_hist'][hist_key] = hist_value * shape_value * correlation_score

        print(f"SumDist for {hist_key}: {scores['sum_shapes_hist'][hist_key]}")


def calc_correlation_score(query, training_set, scores):
    scores['correlation_scores'] = {}

    query_img = query['clipped_img']

    for train in training_set:
        train_img = train['clipped_img']

        min_width = min(query_img.shape[1], train_img.shape[1])
        min_height = min(query_img.shape[0], train_img.shape[0])

        same_dimensions_train = train_img[0:min_height, 0:min_width]
        same_dimensions_query = query_img[0:min_height, 0:min_width]

        corr = skimage.feature.match_template(same_dimensions_query, same_dimensions_train, pad_input=True)
        first_corr = corr.max()

        corr = skimage.feature.match_template(cv2.flip(same_dimensions_query, -1), same_dimensions_train, pad_input=True)
        second_corr = corr.max()

        corr = max(first_corr, second_corr)

        scores['correlation_scores'][train['filename']] = corr
        print(f"CorrScore for {train['filename']}: {scores['correlation_scores'][train['filename']]}")


def calc_corners_score(query, training_set, scores):
    scores['corners_scores'] = {}

    for train in training_set:
        percentage_match, matches = match_corners(query, train)
        print(f"Corners match for {train['filename']}: {percentage_match}")
        scores['corners_scores'][train['filename']] = percentage_match


def calc_histogram_score(query, training_set, scores):
    scores['hist_scores'] = {}

    for train in training_set:
        # For some similarity functions a LARGER value indicates higher similarity (Correlation and Intersection).
        # And for others, a SMALLER value indicates higher similarity (Chi-Squared and Hellinger).
        hist_dist = cv2.compareHist(query['hist'], train['hist'], cv2.HISTCMP_CORREL)
        print(f"HistDist for {train['filename']}: {hist_dist}")
        scores['hist_scores'][train['filename']] = hist_dist


def calc_shapes_score(query, training_set, scores):
    scores['shapes_scores'] = {}

    for train in training_set:
        match_coeff = cv2.matchShapes(train['edges_img'], query['edges_img'], method=1, parameter=0.0)
        print(f"MatchCoeff for {train['filename']}: {match_coeff}")
        scores['shapes_scores'][train['filename']] = match_coeff


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

    if rect:
        rect_text = "Bounding rect ({0:.0f},{1:.0f}), angle {2:.0f}".format(rect[1][0], rect[1][1], rect[2])

        cv2.putText(combined, rect_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        print("Image {0} {1}".format(img_idx, rect_text))

    cv2.imwrite('out/{0}-combined.png'.format(img_idx), combined)


def iterate_over_images_quick_test(img_names):
    for img_idx, img_name in enumerate(img_names):
        res = load_and_prepare_img(img_name)

        find_max_contour(res)
        align_and_clip(res, 'grey_img')
        ensure_max_width(res, 'clipped_img')

        find_good_features_to_track(res, 'clipped_img')
        cv2.imwrite('out/{0}-corners.png'.format(img_idx), res['corners_img'])

        # find_sift(res, 'clipped_img')
        # cv2.imwrite('out/{0}-sift.png'.format(img_idx), res['sift_img'])


if __name__ == '__main__':
    training_filenames = ['in/control/' + file for file in listdir('in/control')]
    training_filenames = [file for file in training_filenames if isfile(file)]

    # training_filenames = [
    #                       'in/control/10.bmp',
    #                       'in/control/13.bmp',
    #                       'in/control/21.bmp',
    #                       'in/control/22.bmp',
    #                       ]

    # training_filenames = [
    #     'in/control/14.bmp'
    # ]

    query_filenames = ['in/trial02/' + file for file in listdir('in/trial02')]
    query_filenames = [file for file in query_filenames if isfile(file)]

    query_filenames = [
        'in/trial02/04.bmp',
    ]

    # iterate_sift(training_filenames, query_filenames)
    detection(training_filenames, query_filenames)
    # calibrate()

    # camera_matrix, dist_coefs = load_calibration()
    # undistort_images(camera_matrix, dist_coefs, glob.glob('in/trial/*.bmp'))

    # iterate_over_images_quick_test(training_filenames)

cv2.destroyAllWindows()