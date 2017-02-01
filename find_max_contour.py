import cv2
import numpy as np


def supress_background_noise(res, img_key):
    img = res[img_key]
    ret, threshold_img = cv2.threshold(img, thresh=240, maxval=255, type=cv2.THRESH_BINARY_INV)
    img[threshold_img == 0] = 255
    return res


def find_edges(res, img_key):
    img = res[img_key]

    # Gausian removes small edges during Canny
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # threshold_img = cv2.bilateralFilter(threshold_img.copy(), dst=threshold_img, d=3, sigmaColor=250, sigmaSpace=250)
    # threshold_img = cv2.erode(threshold_img.copy(), kernel=np.ones((1,1), np.uint8), iterations=2)
    # threshold_img = cv2.morphologyEx(threshold_img.copy(), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # 0, 250, 3, False goes well with the Gaus filter
    # edges_img = cv2.Canny(img, threshold1=0, threshold2=250, apertureSize=3, L2gradient=False)

    edges_img = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=3, L2gradient=True)
    res['edges_img'] = edges_img

    return res


def find_edges_better(res, img_key):
    img = res[img_key]
    ret, threshold_img = cv2.threshold(img, thresh=-1, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    edges_img = cv2.Canny(threshold_img, threshold1=100, threshold2=255, apertureSize=3)

    res['edges_img'] = edges_img

    return res


def find_max_contour(res):
    ret, threshold_img = cv2.threshold(res['grey_img'], thresh=-1, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

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
    else:
        print(f"Warning! No countour in find_max_contour! filename: {res['filename']}")

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

    final_topleft_x = max(int(topleft_x-3), 0)
    final_topleft_y = max(int(topleft_y-3), 0)

    final_bottomright_x = min(int(topleft_x+rect_width+3), width)
    final_bottomright_y = min(int(topleft_y+rect_height+3), height)

    res['clipped_img'] = img[final_topleft_y:final_bottomright_y,
                             final_topleft_x:final_bottomright_x]

    return res


def ensure_max_width(res, img_key):
    img = res[img_key]

    if img.shape[0] > img.shape[1]:
        res[img_key] = cv2.transpose(img)

    return res