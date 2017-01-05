import cv2
import numpy as np


def find_max_contour(res):
    ret, threshold_img = cv2.threshold(res['grey_img'], thresh=210, maxval=255, type=cv2.THRESH_BINARY_INV)

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
    contour_rect = None
    bounding_box = None
    max_contour = None

    if len(contours) != 0:
        areas = [cv2.contourArea(contour) for contour in contours]
        max_area_index = np.argmax(areas)

        # perimeter_of_max_area = cv2.arcLength(contours[max_area_index], True)
        # perimeters = [cv2.arcLength(contour, True) for contour in contours]
        # max_perimeter_index = np.argmax(perimeters)
        # area_of_max_perimeter = cv2.contourArea(contours[max_perimeter_index])

        max_contour = contours[max_area_index]
        bounding_box = cv2.boundingRect(max_contour)
        contour_rect = cv2.minAreaRect(max_contour)

        # Generate contours image
        contours_img = np.zeros(threshold_img.shape)

        levels = []
        level_colors = [(255-idx*10, 255-idx*10, 255-idx*10) for idx in range(0, 25)]
        curr_level = 0
        while curr_level < len(level_colors):
            levels.append([iter_contour for idx, iter_contour in enumerate(contours) if hierarchy[0][idx][3] == curr_level-1])
            curr_level += 1
        for level_idx, level in enumerate(levels):
            cv2.drawContours(contours_img, contours=levels[level_idx], contourIdx= -1, color=level_colors[level_idx], thickness=1)

        cv2.drawContours(contours_img, contours=contours, contourIdx= max_area_index, color=255, thickness=2)

        box = cv2.boxPoints(contour_rect)
        cv2.drawContours(contours_img, [np.array(box).astype(int)], 0, (200, 200, 200), thickness=2)

    res['max_contour'] = max_contour
    res['all_contours'] = contours
    res['contour_rect'] = contour_rect
    res['bounding_box'] = bounding_box
    res['threshold_img'] = threshold_img
    res['edges_img'] = edges_img
    res['contours_img'] = contours_img

    return res


def find_moments(res, img_key, binary_image):
    img = res[img_key]

    res['moments'] = cv2.moments(img, binaryImage=binary_image)
    res['hu_moments'] = cv2.HuMoments(res['moments'])

    return res


def clip_img(res, img_key):
    img = res[img_key]
    bounding_box = res['bounding_box']

    res['clipped_img'] = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                             bounding_box[0]:bounding_box[0] + bounding_box[2]]

    return res


def align_and_clip(res, img_key):
    img = res[img_key]
    width,height = img.shape

    rect_width = res['contour_rect'][1][0]
    rect_height = res['contour_rect'][1][1]
    angle = res['contour_rect'][2]

    rot_center_x = res['contour_rect'][0][0]
    rot_center_y = res['contour_rect'][0][1]

    M = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle, 1)
    img = cv2.warpAffine(img, M, (width, height))

    topleft_x = rot_center_x - (rect_width*0.5)
    topleft_y = rot_center_y - (rect_height*0.5)

    res['clipped_img'] = img[int(topleft_y):int(topleft_y+rect_height),
                             int(topleft_x):int(topleft_x+rect_width)]

    # _, res['clipped_threshold_img'] = cv2.threshold(res['clipped_img'], thresh=180, maxval=255,
    #                                                 type=cv2.THRESH_BINARY_INV)
    otsu_threshold, res['clipped_threshold_img'] = cv2.threshold(res['clipped_img'], thresh=0, maxval=255,
                                                                 type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return res


def save_to_file_max_contour(res, img_idx):
    img_01 = res['contours_img']
    img_02 = res['grey_img']
    img_03 = res['threshold_img']
    img_04 = res['edges_img']

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
