import cv2
import numpy as np
import pandas as pd

import Localization
import Recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""

def CaptureFrame_Process(file_path, sample_frequency):
    plates = localize_and_recognize(file_path, sample_frequency) # Output is ("XX99XX", 1, 1/sf)
    groups = scene_change_separate(plates, sample_frequency, threshold=3, frame_distance = 50)
    plate_strings = voting_per_letter(groups)
    result = []
    for plate_string in plate_strings:
        plate_new = dutch_check(plate_string[0])
        if plate_new != '':
          result.append((plate_new, plate_string[1], plate_string[2]))
    return result

def localize_and_recognize(file_path, sample_frequency):
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    result = []
    frame_number = 1
    while ret:
        frame_number += 1
        frame = np.array(frame)
        plates = Localization.plate_detection(frame)
        if plates is None:
            ret, frame = cap.read()
            continue
        for plate in plates:
            if np.any(plate):
                letters = Recognize.segment_and_recognize(plate)
                if letters != '':
                    result.append((letters, frame_number, frame_number/sample_frequency))
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return result
def voting_consecutive(plates, indices, sample_frequency):
    result = []
    for start,end in indices:
        count = {}
        for tupl in plates[start:end]:
            plate = tupl[0]
            if plate in count:
                count[plate] += 1
            else:
                count[plate] = 1
        average_frame_number = (plates[start][1] + plates[end-1][1]) // 2
        result.append((max(count, key=count.get), average_frame_number, average_frame_number/sample_frequency))
    return result
def scene_change_consecutive(plates, threshold = 2, num_avg = 7):
    last_plates = []
    indices = []
    start_idx = 0
    for i in range(len(plates)):
        plate = plates[i][0]
        if len(last_plates) < num_avg:
            last_plates.append(plate)
            continue
        average = 0
        for last in last_plates:
            average += hamming_dist(last, plate)
        average /= num_avg
        if average > threshold:
            end_idx = i
            indices.append((start_idx, end_idx))
            start_idx = i+1
            last_plates = []
    indices.append((start_idx, len(plates)))
    return indices

def voting_separate(groups):
    result = []
    for group in groups:
        count = {}
        for plate in group[0]:
            plate = plate[0]
            if plate in count:
                count[plate] += 1
            else:
                count[plate] = 1
        result.append((max(count, key=count.get), group[1], group[2]))
    return result
def scene_change_separate(plates, sample_frequency, threshold=3, frame_distance = 50):
    plates_copy = []
    for plate in plates:
        plates_copy.append(plate)

    groups = []
    while len(plates_copy) > 0:
        first_plate = plates_copy[0]
        num_plates = 1
        num_frame = first_plate[1]
        group = [first_plate]
        plates_copy.remove(first_plate)
        for plate in plates_copy:
            if hamming_dist(first_plate[0], plate[0]) <= threshold and plate[1] - first_plate[1] <= frame_distance:
                num_plates += 1
                num_frame += plate[1]
                group.append(plate)
        for plate in group:
            if plate in plates_copy:
                plates_copy.remove(plate)
        avg_frame = num_frame//num_plates
        groups.append((group, avg_frame, avg_frame/sample_frequency))
    return groups

def voting_per_letter(groups):
    result = []
    for group in groups:
        new_letters = ''
        for i in range(6):
            count = {}
            for plate in group[0]:
                if len(plate[0]) != 6:
                    continue
                letter = plate[0][i]
                if letter in count:
                    count[letter] += 1
                else:
                    count[letter] = 1
            if len(count) > 0:
                new_letters += max(count, key=count.get)
        result.append((new_letters, group[1], group[2]))
    return result

def hamming_dist(s1, s2):
    dist = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            dist += 1
    dist += max(len(s1), len(s2)) - min(len(s1), len(s2))
    return dist

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