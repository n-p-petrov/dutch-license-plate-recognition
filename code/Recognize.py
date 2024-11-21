import cv2
import numpy as np
import os

import Commons

"""
In this file, you will define your own segment_and_recognize function.
To do:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
    3. Recognize the character by comparing the distances
Inputs:(One)
    1. plate_imgs: cropped plate images by Localization.plate_detection function
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
    1. recognized_plates: recognized plate characters
    type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
    You may need to define other functions.
"""
def segment_and_recognize(plate_img, threshold=0.08, mean_c=15, xor_ratio=1, dash_ratio = 0.3):
    path = 'dataset/LettersNumbers/'
    img = threshold_and_denoise(plate_img, mean_c)
    indexes = segment(img, threshold, dash_ratio)
    letters = xor(indexes, xor_ratio, path)
    #letters = dutch_check(letters)
    return letters

    # SIFT through the images
    # letters = ""
    # for image in indexes:
    #     show(image)
    #     image = cv2.resize(image, (32, 32))
    #     distance = {}
    #     des = sift_descriptor_32(image)
    #     for descriptor in database:
    #         dist = np.linalg.norm(database[descriptor] - des)
    #         distance[descriptor] = dist
    #     letters += min(distance, key=distance.get)
    #     print(min(distance, key=distance.get), distance)
    # print(letters)
    # show(img_thresh)
    # RP-NL-93
    # RP-NLH98

def dutch_check(letters):
    new_letters = ''

    if len(letters) != 6:
        return ''
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digit = '1234567890'

    plate = []
    for letter in letters:
        if letter in alphabet:
            plate.append(True)
        elif letter in digit:
            plate.append(False)

    #XX-99-XX
    #99-XX-99
    #XX-XX-99
    #99-XX-XX

    if np.array_equal(plate,[True, True, False, False, True, True])\
            or np.array_equal(plate, [False, False, True, True, False, False])\
            or np.array_equal(plate, [False, False, False, False, True, True])\
            or np.array_equal(plate, [True, True, False, False, False, False])\
            or np.array_equal(plate, [True, True, True, True, False, False])\
            or np.array_equal(plate, [False, False, True, True, True, True]):
        new_letters = letters[0] + letters[1] + "-" + letters[2] + letters[3] + "-" + letters[4] + letters[5]
    #99-XXX-9
    #XX-999-X
    elif plate == [False, False, True, True, True, False]\
            or plate == [True, True, False, False, False, True]:
        new_letters = letters[0] + letters[1] + "-" + letters[2] + letters[3] + letters[4] + "-" + letters[5]
    #9-XXX-99
    #X-999-XX
    elif    plate == [False, True, True, True, False, False]\
            or plate == [True, False, False, False, True, True]:
        new_letters = letters[0] + "-" + letters[1] + letters[2] + letters[3] + "-" + letters[4] + letters[5]
    #XXX-99-X
    #999-XX-9
    elif plate == [False, False, False, True, True, False]\
            or plate == [True, True, True, False, False, True]:
        new_letters = letters[0] + letters[1] + letters[2] + "-" + letters[3] + letters[4] + "-" + letters[5]
    else:
        return ''
    return new_letters



def threshold_and_denoise(image, mean_c):
    img_big = cv2.resize(image, (2 * image.shape[1], 2 * image.shape[0]))
    img_gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
    # men = np.min(img_gray)
    # mex = np.max(img_gray)
    # for i in range(img_gray.shape[0]):
    #     for j in range(img_gray.shape[1]):
    #         img_gray[i][j] = ((img_gray[i][j] - men) / (mex - men)) * 255.
    ksize = img_gray.shape[0] if img_gray.shape[0] % 2 == 1 else img_gray.shape[0] + 1
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       ksize, mean_c)

    # denoise the image
    ksize = 3
    img_erode = cv2.erode(img_thresh, np.ones([ksize, ksize]))
    img_denoise = cv2.dilate(img_erode, np.ones([ksize, ksize]))
    return img_denoise
def segment(image, threshold, dash_ratio):
    started = False
    indexes = []
    for col in range(image.shape[1]):
        pillar = image[:, col]
        total_white = len(pillar[pillar == 255])
        ratio = total_white / len(pillar)
        if not started and ratio >= threshold:     # Crop the columns
            start_col = col
            started = True
        if started and ratio < threshold:
            started = False
            end_col = col
            for row in range(0, image.shape[0]): # Crop the rows
                pillar = image[row, :]
                total_white = len(pillar[pillar == 255])
                ratio = total_white / len(pillar)
                if not started and ratio >= threshold:
                    start_row = row
                    started = True
                if started and ratio < threshold:
                    end_row = row
                    letter = image[start_row:end_row, start_col:end_col]
                    if np.count_nonzero(letter)/(letter.shape[0] * letter.shape[1]) > dash_ratio and end_col - start_col > 5:
                        indexes.append(letter)
                    started = False
                    break
    return indexes
def xor(indexes, xor_ratio, path):
    letters = ''
    for image in indexes:
        xor_distance = {}
        image = cv2.resize(image, (32, 32))
        image[image > 0] = 255
        for fname in os.listdir(path):
            if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
                ltr = cv2.imread(path + fname)
                ltr = cv2.cvtColor(ltr, cv2.COLOR_BGR2GRAY)
                for col in range(ltr.shape[1]):
                    all_zeros = not np.any(ltr[:, col])
                    if all_zeros:
                        ltr = ltr[:, 0:col]
                        break
                ltr = cv2.resize(ltr, (32, 32))
                ltr[ltr > 0] = 255

                xor_distance[fname.split(".")[0]] = np.count_nonzero(np.logical_xor(image, ltr))

        first_letter = min(xor_distance, key=xor_distance.get)
        first = xor_distance.pop(first_letter)
        second = xor_distance.pop(min(xor_distance, key=xor_distance.get))
        accuracy = first / second
        if accuracy <= xor_ratio:
            letters += first_letter
    return letters

def get_color_segmentation_mask(image):
    color_min = np.array([0, 0, 0])
    color_max = np.array([255, 255, 100])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, color_min, color_max)
    #show_cv(mask)
    return mask

def show_plt(title, img):
    Commons.plotImage(img, title, cmapType="gray")

def show_cv(title,img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_database(directory, filename):
    file = open(directory + filename, 'w')
    for fname in os.listdir(directory):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread(directory + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for col in range(image.shape[1]):
                all_zeros = not np.any(image[:, col])
                if all_zeros:
                    image = image[:, 0:col]
                    break
            image = cv2.resize(image, (32, 32))
            sift = sift_descriptor_32(image)
            file.write(fname.split('.')[0] + '|' + ', '.join([str(res) for res in sift]) + '\n')
    file.close()

def read_database(filename):
    file = open(filename, 'r')
    result = {}
    for line in file.read().split('\n'):
        if len(line) < 1:
            continue
        line_info = line.split('|')
        result[line_info[0]] = np.array([float(a) for a in line_info[1].split(', ')])
    file.close()
    return result

# def sift_descriptor(image_interest_patch):
#     result = np.zeros(128)
#
#     # Make sure the size of the image_interest_patch is divisible by 16*16
#     Sobel_kernel_y = np.zeros((3, 3))
#
#     Sobel_kernel_y[0, :] = [-1, -2, -1]
#     Sobel_kernel_y[2, :] = [1, 2, 1]
#
#     Sobel_kernel_x = np.zeros((3, 3))
#
#     Sobel_kernel_x[:, 0] = [-1, -2, -1]
#     Sobel_kernel_x[:, 2] = [1, 2, 1]
#
#     img_x = np.float64(image_interest_patch)
#     img_y = np.float64(image_interest_patch)
#     I_x = cv2.filter2D(img_x, -1, Sobel_kernel_x)
#     I_y = cv2.filter2D(img_y, -1, Sobel_kernel_y)
#
#     I_x[I_x == 0] = 0.0001
#     gradient_orientation = np.arctan(I_y, I_x)
#     gradient_magnitude = np.hypot(I_x, I_y)
#
#     # find big histogram for principal direction
#     hist = np.histogram(gradient_orientation, bins=8, weights=gradient_magnitude)[0]
#
#     peak = np.where(hist == max(hist))[0]
#
#     # calculate histogram of each box
#     bins = 0
#     for x in range(0, image_interest_patch.shape[0], 4):
#         for y in range(0, image_interest_patch.shape[1], 4):
#             if bins >= 128:
#                 break
#             box_hist = np.histogram(gradient_orientation[x: x + 4, y: y + 4], bins=8,
#                                     weights=gradient_magnitude[x:x + 4, y:y + 4])[0]
#             result[bins:bins + 8] = np.roll(box_hist, -peak)
#             bins += 8
#
#     # normalise result
#     result = result / np.linalg.norm(result)
#     return result

def sift_descriptor_32(image_interest_patch):
    result = np.zeros(128)

    # Make sure the size of the image_interest_patch is divisible by 16*16
    Sobel_kernel_y = np.zeros((3, 3))

    Sobel_kernel_y[0, :] = [-1, -2, -1]
    Sobel_kernel_y[2, :] = [1, 2, 1]

    Sobel_kernel_x = np.zeros((3, 3))

    Sobel_kernel_x[:, 0] = [-1, -2, -1]
    Sobel_kernel_x[:, 2] = [1, 2, 1]

    img_x = np.float64(image_interest_patch)
    img_y = np.float64(image_interest_patch)
    I_x = cv2.filter2D(img_x, -1, Sobel_kernel_x)
    I_y = cv2.filter2D(img_y, -1, Sobel_kernel_y)

    I_x[I_x == 0] = 0.0001
    gradient_orientation = np.arctan(I_y, I_x)
    gradient_magnitude = np.hypot(I_x, I_y)

    # find big histogram for principal direction
    hist = np.histogram(gradient_orientation, bins=8, weights=gradient_magnitude)[0]

    peak = np.where(hist == max(hist))[0]

    # calculate histogram of each box
    bins = 0
    for x in range(0, image_interest_patch.shape[0], 8):
        for y in range(0, image_interest_patch.shape[1], 8):
            if bins >= 128:
                break
            box_hist = np.histogram(gradient_orientation[x: x + 8, y: y + 8], bins=8,
                                    weights=gradient_magnitude[x:x + 8, y:y + 8])[0]
            result[bins:bins + 8] = np.roll(box_hist, -peak)
            bins += 8

    # normalise result
    result = result / np.linalg.norm(result)
    return result