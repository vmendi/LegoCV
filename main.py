from time import strftime

import cv2
import numpy as np

import video


# http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html


def extract_max_contour_rect(img):
    ret, threshold_img = cv2.threshold(img, thresh=200, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Gausian removes small edges during Canny
    threshold_img = cv2.GaussianBlur(threshold_img, (11, 11), 0)
    # threshold_img = cv2.bilateralFilter(threshold_img.copy(), dst=threshold_img, d=3, sigmaColor=250, sigmaSpace=250)
    # threshold_img = cv2.erode(threshold_img.copy(), kernel=np.ones((1,1), np.uint8), iterations=2)
    # threshold_img = cv2.morphologyEx(threshold_img.copy(), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    edges_img = cv2.Canny(threshold_img, threshold1=100, threshold2=255, apertureSize=3)

    # Close before findContours helps with discontinuities in the perimeters
    edges_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Extract contours
    contours_img, contours, hierarchy = cv2.findContours(edges_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_area_contour_rect = None

    if len(contours) != 0:
        areas = [cv2.contourArea(contour) for contour in contours]
        max_area_index = np.argmax(areas)
        perimeter_of_max_area = cv2.arcLength(contours[max_area_index], True)

        perimeters = [cv2.arcLength(contour, True) for contour in contours]
        max_perimeter_index = np.argmax(perimeters)
        area_of_max_perimeter = cv2.contourArea(contours[max_perimeter_index])

        # Generate contours image
        contours_img = np.zeros(threshold_img.shape)

        levels = []
        level_colors = [(255-idx*10, 255-idx*10, 255-idx*10) for idx in range(0, 25)]
        curr_level = 0
        while curr_level < len(level_colors):
            levels.append([contour for idx, contour in enumerate(contours) if hierarchy[0][idx][3] == curr_level-1])
            curr_level += 1
        for level_idx, level in enumerate(levels):
            cv2.drawContours(contours_img, contours=levels[level_idx], contourIdx= -1, color=level_colors[level_idx], thickness=1)

        max_area_contour_rect = cv2.minAreaRect(contours[max_area_index])
        box = cv2.boxPoints(max_area_contour_rect)
        cv2.drawContours(contours_img, [np.array(box).astype(int)], 0, (200, 200, 200), thickness=2)

    return {'max_area_contour_rect': max_area_contour_rect,
            'threshold': threshold_img,
            'edges': edges_img,
            'contours': contours_img}


def save_to_file(detection_result, img_idx):
    cv2.imwrite('out/{0}-threshold.png'.format(img_idx), detection_result['threshold'])
    cv2.imwrite('out/{0}-edges.png'.format(img_idx), detection_result['edges'])
    cv2.imwrite('out/{0}-contours.png'.format(img_idx), detection_result['contours'])

    rect = detection_result['rect']

    print("Image {0} bounding rect ({1:.0f},{2:.0f}), angle {3:.0f}".format(img_idx, rect[1][0], rect[1][1], rect[2]))


def iterate_over_images_detection(img_names):
    for img_idx, img_name in enumerate(img_names):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (1024, 768))

        height, width, color_depth = img.shape
        img = img[50:height-50, 150:width-150]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detection_result = extract_max_contour_rect(gray)
        save_to_file(detection_result, img_idx)
        # find_corners(img_idx, img, gray)


def realtime_detection():
    cap = video.create_capture(1)

    while True:
        flag, captured_img = cap.read()

        height, width, color_depth = captured_img.shape
        captured_img = captured_img[100:height-100, 400:width-400]

        gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
        detection_result = extract_max_contour_rect(gray)

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

        cv2.imshow('main_00', cv2.resize(captured_img_00, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
        cv2.imshow('main_01', cv2.resize(captured_img_01, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.imwrite('in/{0}_00.png'.format(strftime("%Y-%m-%d %H-%M-%S")), captured_img_00)
            cv2.imwrite('in/{0}_01.png'.format(strftime("%Y-%m-%d %H-%M-%S")), captured_img_01)


if __name__ == '__main__':

    filenames = ['in/IMG_00064.jpg', 'in/IMG_00065.jpg', 'in/IMG_00066.jpg', 'in/IMG_00068.jpg', 'in/IMG_00070.jpg',
                 'in/IMG_00072.jpg', 'in/IMG_00073.jpg', 'in/IMG_00074.jpg', 'in/IMG_00075.jpg', 'in/IMG_00076.jpg',
                 'in/IMG_00077.jpg', 'in/IMG_00078.jpg']

    # iterate_over_images_detection(filenames)
    # realtime_detection()
    capture()

cv2.destroyAllWindows()