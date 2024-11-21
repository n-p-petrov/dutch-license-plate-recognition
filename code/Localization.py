import cv2
import numpy as np
from collections import deque

import Commons

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(image):
    bounding_boxes = localize_plates(image)
    plates = crop_with_offset(image, bounding_boxes, 3)
    plates = normalize_rotation(plates)
    if plates is None:
        return
    for plate in plates:
        if plate.shape[0]*plate.shape[1] < 50:
            plates.remove(plate)
    return plates


def normalize_rotation(license_plates):

    normalized_plates = []

    for plate in license_plates:
        #Commons.plotImage(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), "before rotation")
        edge_image = get_edge_image(plate)

        lines = get_Hough_lines(edge_image)
        if lines is None:
            return
        copy = np.array(plate)
        # Commons.draw_Hough_lines(copy, lines)
        # Commons.plotImage(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB), "hough")
        theta = lines[0, 0, 1]
        plate = rotate_image(plate, theta)

        bounding_box = localize_plate(plate)
        plate = crop_with_offset(plate, bounding_box, 0)[0]
        if plate.any():
            # Commons.plotImage(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), "after rotation")
            # cv2.imshow("title", cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            normalized_plates.append(plate)

    return normalized_plates


def get_edge_image(image):
    upper_threshold = 150
    lower_threshold = 130

    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(gs_image, upper_threshold, lower_threshold, apertureSize=3)


def get_Hough_lines(edges):
    rho_step_size = 1
    theta_step_size = np.pi / 90
    min_votes = 50

    return cv2.HoughLines(edges, rho_step_size, theta_step_size, min_votes)


def rotate_image(image, theta):
    rows = image.shape[0]
    cols = image.shape[1]
    center_of_rotation = (cols / 2, rows / 2)
    angle_of_rotation = (theta - np.pi / 2) / np.pi * 180
    scaling_factor = 1
    rotation_matrix = cv2.getRotationMatrix2D(center_of_rotation, angle_of_rotation, scaling_factor)

    return cv2.warpAffine(image, rotation_matrix, (cols, rows))


def localize_plates(image):
    mask = get_color_segmentation_mask(image)
    mask = closing(mask)
    bounding_boxes = get_bounding_boxes(mask)

    return bounding_boxes


def get_bounding_box(mask):
    rows = mask.shape[0]
    cols = mask.shape[1]

    min_y = cols
    min_x = rows
    max_y = 0
    max_x = 0

    for row in range(rows):
        for col in range(cols):
            if mask[row, col]:
                min_y = min(min_y, row)
                min_x = min(min_x, col)

                max_y = max(max_y, row)
                max_x = max(max_x, col)

    return [[min_x, min_y, max_x, max_y]]


def localize_plate(image):
    # edge_image = get_edge_image(image)
    # Commons.plotImage(edge_image, "plate edges", "gray")
    mask = get_color_segmentation_mask(image)
    mask = denoise_plate_mask(mask)
    # Commons.plotImage(mask, "plate mask", "gray")
    bounding_box = get_bounding_box(mask)

    return bounding_box


# mask[i, j] = 0 if pixel underneath is not yellow
# mask[i, j] = 255 if pixel underneath is yellow
def get_color_segmentation_mask(image):
    color_min = np.array([10, 90, 90])
    color_max = np.array([45, 255, 255])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, color_min, color_max)
    return mask


def closing(mask):
    mask_width = mask.shape[1]
    kernel_width = int(0.02 * mask_width)
    kernel = np.ones((kernel_width, kernel_width))

    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    return mask


def denoise_plate_mask(mask):
    mask_width = mask.shape[1]

    kernel_width = int(0.02 * mask_width)
    kernel = np.ones((kernel_width, kernel_width))

    mask = cv2.dilate(mask, kernel)

    kernel_width = int(0.06 * mask_width)
    kernel = np.ones((kernel_width, kernel_width))

    mask = cv2.erode(mask, kernel)

    return mask


def get_bounding_boxes(mask):
    bounding_boxes = []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = cv2.drawContours(np.zeros(mask.shape), contours, -1, (255, 255, 0), 3)
    # Commons.plotImage(mask, "mask")

    for contour in contours:
        min_y, min_x = mask.shape
        max_x = 0
        max_y = 0
        for point in contour:
            x, y = point[0]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        bounding_box = [min_x, min_y, max_x, max_y]
        if isValid(bounding_box):
            bounding_boxes.append([min_x, min_y, max_x, max_y])

    return bounding_boxes


def isValid(bounding_box):
    # TODO: think about more conditions
    min_plate_width = 60

    height = bounding_box[3] - bounding_box[1]
    width = bounding_box[2] - bounding_box[0]

    if width > min_plate_width and height < .6 * width and height > 5:
        return True

    return False


def crop_with_offset(image, bounding_boxes, offset=10):
    cropped_images = []
    height = image.shape[0]
    width = image.shape[1]

    for bounding_box in bounding_boxes:
        min_x, min_y, max_x, max_y = bounding_box

        start_row = max(min_y - offset, 0)
        end_row = min(max_y + offset, height - 1)

        start_col = max(min_x - offset, 0)
        end_col = min(max_x + offset, width - 1)

        cropped_images.append(image[start_row:end_row, start_col:end_col])

    return cropped_images
