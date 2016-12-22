import cv2
import numpy as np


def find_max_contour(img):
    ret, threshold_img = cv2.threshold(img, thresh=230, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Gausian removes small edges during Canny
    # threshold_img = cv2.GaussianBlur(threshold_img, (11, 11), 0)
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
            'contours': contours_img,
            'original': img}