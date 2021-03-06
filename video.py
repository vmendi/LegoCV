import cv2
import time

from find_max_contour import find_max_contour


def realtime_detection_quick_structure():
    cap = create_capture(1)

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


def capture_from_two_cameras():
    cap_cam00 = create_capture(1)
    cap_cam01 = create_capture(2)

    while True:
        flag, captured_img_00 = cap_cam00.read()
        flag, captured_img_01 = cap_cam01.read()

        if captured_img_00 is None or captured_img_01 is None:
            continue

        cv2.imshow('main_00', captured_img_00)
        cv2.imshow('main_01', cv2.resize(captured_img_01, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.imwrite('in/{0}_00.png'.format(time.strftime("%Y-%m-%d %H-%M-%S")), captured_img_00)
            cv2.imwrite('in/{0}_01.png'.format(time.strftime("%Y-%m-%d %H-%M-%S")), captured_img_01)



#   0  CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
#   1  CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#   2  CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
#   3  CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#   4  CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#   5  CV_CAP_PROP_FPS Frame rate.
#   6  CV_CAP_PROP_FOURCC 4-character code of codec.
#   7  CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#   8  CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#   9 CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#   10 CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#   11 CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#   12 CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#   13 CV_CAP_PROP_HUE Hue of the image (only for cameras).
#   14 CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#   15 CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#   16 CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#   17 CV_CAP_PROP_WHITE_BALANCE Currently unsupported
#   18 CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

def create_capture(source = 0):
    cap = cv2.VideoCapture(source)

    # Change the camera setting using the set() function
    # cap.set(cv2.CAP_PROP_EXPOSURE, 10)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.cv.CV_CAP_PROP_GAIN, 4.0)
    # cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 144.0)
    # cap.set(cv2.CAP_PROP_CONTRAST, 27.0)
    # cap.set(cv2.cv.CV_CAP_PROP_HUE, 13.0) # 13.0
    # cap.set(cv2.cv.CV_CAP_PROP_SATURATION, 28.0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592.0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944.0)

    # cap.set(cv2.CAP_PROP_FPS, 5.0)

    # cap.set(cv2.CAP_PROP_FORMAT, cv2.COLOR_YUV2RGB_YUY2)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)


    # Read the current setting from the camera
    test = cap.get(cv2.CAP_PROP_POS_MSEC)
    ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    img_format = cap.get(cv2.CAP_PROP_FORMAT)
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    auto_focus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    print("Test: ", test)
    print("Ratio: ", ratio)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    print("Brightness: ", brightness)
    print("Contrast: ", contrast)
    print("Saturation: ", saturation)
    print("Hue: ", hue)
    print("Gain: ", gain)
    print("Exposure: ", exposure)
    print("Auto-Exposure: ", auto_exposure)
    print("Format: ", img_format)
    print("Focus: ", focus)
    print("Auto-Focus: ", auto_focus)
    print("FourCC: ", fourcc)

    return cap