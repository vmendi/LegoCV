from os.path import basename
from os.path import splitext

import numpy as np
import cv2
import glob


def calibrate():
    square_size = 1.0

    pattern_size = (9, 6)
    img_names = glob.glob('in/calibration/blah*.bmp')

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0

    for filename in img_names:
        print('processing %s... ' % filename, end='')
        img = cv2.imread(filename, 0)

        h, w = img.shape[:2]
        patter_was_found, corners = cv2.findChessboardCorners(img, pattern_size)
        if patter_was_found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_color, pattern_size, corners, patter_was_found)
            only_file_name = splitext(basename(filename))[0]
            outfile = f'out/calibration/{only_file_name}_found.png'
            cv2.imwrite(outfile, img_color)

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)
            print('done')
        else:
            print('chessboard not found')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # root mean square (RMS) reprojection error should be between 0.1 and 1.0 pixels in a good calibration.
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    print_error(camera_matrix, dist_coefs, img_points, obj_points, rvecs, tvecs)
    undistort_images(camera_matrix, dist_coefs, glob.glob('in/calibration/*.bmp'))

    np.savez('out/calibration/calibration.npz', camera_matrix, dist_coefs)


def load_calibration():
    npzfile = np.load('out/calibration/calibration.npz')

    camera_matrix = npzfile['arr_0']
    dist_coefs = npzfile['arr_1']

    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    return camera_matrix, dist_coefs

def undistort_images(camera_matrix, dist_coefs, img_names_undistort):

    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # ...and save the undistorted image
        only_file_name = splitext(basename(img_found))[0]
        outfile = f'out/calibration/{only_file_name}_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)


# RMS is already the reprojection error (sigh, we reinvented the wheel)
def print_error(camera_matrix, dist_coefs, img_points, obj_points, rvecs, tvecs):
    tot_error = 0
    tot_points = 0
    per_view_errors = []
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
        error = cv2.norm(img_points[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2)

        num_points = len(obj_points[i])
        per_view_errors.append(np.sqrt(error * error / num_points))
        tot_error += error * error
        tot_points += num_points
    print("total error: ", np.sqrt(tot_error / tot_points))