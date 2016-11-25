import cv2
import numpy as np

# http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html


def extract_max_contour_rect(img_idx, img):
    ret, threshold_img = cv2.threshold(img, thresh=128, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Gausian removes small edges during Canny
    threshold_img = cv2.GaussianBlur(threshold_img, (11, 11), 0)
    # threshold_img = cv2.bilateralFilter(threshold_img.copy(), dst=threshold_img, d=3, sigmaColor=250, sigmaSpace=250)
    # threshold_img = cv2.erode(threshold_img.copy(), kernel=np.ones((1,1), np.uint8), iterations=2)
    # threshold_img = cv2.morphologyEx(threshold_img.copy(), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    edges_img = cv2.Canny(threshold_img, threshold1=100, threshold2=255, apertureSize=3)

    # Close before findContours helps with discontinuities in the perimeters
    edges_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours_img, contours, hierarchy = cv2.findContours(edges_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(contour) for contour in contours]
    max_area_index = np.argmax(areas)
    perimeter_of_max_area = cv2.arcLength(contours[max_area_index], True)

    perimeters = [cv2.arcLength(contour, True) for contour in contours]
    max_perimeter_index = np.argmax(perimeters)
    area_of_max_perimeter = cv2.contourArea(contours[max_perimeter_index])

    contours_img = np.zeros(threshold_img.shape)

    levels = []
    level_colors = [(255-idx*10, 255-idx*10, 255-idx*10) for idx in range(0, 25)]
    curr_level = 0
    while curr_level < len(level_colors):
        levels.append([contour for idx, contour in enumerate(contours) if hierarchy[0][idx][3] == curr_level-1])
        curr_level += 1
    for level_idx, level in enumerate(levels):
        cv2.drawContours(contours_img, contours=levels[level_idx], contourIdx= -1, color=level_colors[level_idx], thickness=1)

    rect = cv2.minAreaRect(contours[max_area_index])
    box = cv2.boxPoints(rect)
    cv2.drawContours(contours_img, [np.array(box).astype(int)], 0, (200, 200, 200), thickness=2)

    cv2.imwrite('out/{0}-threshold.png'.format(img_idx), threshold_img)
    cv2.imwrite('out/{0}-edges.png'.format(img_idx), edges_img)
    cv2.imwrite('out/{0}-contours.png'.format(img_idx), contours_img)

    print("Image {0} bounding rect ({1:.0f},{2:.0f}), angle {3:.0f}".format(img_idx, rect[1][0], rect[1][1], rect[2]))

    return rect


def find_corners(img_idx, img, gray):
    img = img.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01*dst.max()]= [0, 0, 255]

    cv2.imwrite('out/{0}-corners.png'.format(img_idx), img)


def find_corners_subpixel(img_idx, img, gray):
    img = img.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=10, k=0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int8(res)
    img[res[:, 1],res[:, 0]] = [0, 0, 255]
    img[res[:, 3],res[:, 2]] = [0, 255, 0]

    cv2.imwrite('out/{0}-corners.png'.format(img_idx), img)


def iterate_over_images(img_names):
    for img_idx, img_name in enumerate(img_names):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (1024, 768))

        height, width, color_depth = img.shape
        img = img[50:height-50, 150:width-150]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        extract_max_contour_rect(img_idx, gray)
        find_corners(img_idx, img, gray)

if __name__ == '__main__':
    cv2.namedWindow('main', flags=cv2.WINDOW_NORMAL)

    filenames = ['in/IMG_00064.jpg', 'in/IMG_00065.jpg', 'in/IMG_00066.jpg', 'in/IMG_00068.jpg', 'in/IMG_00070.jpg',
                 'in/IMG_00072.jpg', 'in/IMG_00073.jpg', 'in/IMG_00074.jpg', 'in/IMG_00075.jpg', 'in/IMG_00076.jpg',
                 'in/IMG_00077.jpg', 'in/IMG_00078.jpg']

    iterate_over_images(filenames)

cv2.destroyAllWindows()