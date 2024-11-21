import cv2
import os
import csv

import numpy as np

import Commons
import Localization


def calculate_area_bounding_box(bounding_box):
    if bounding_box[0] >= bounding_box[2] or bounding_box[1] >= bounding_box[3]:
        return 0

    height = bounding_box[3] - bounding_box[1]
    width = bounding_box[2] - bounding_box[0]

    return width * height


def evaluate_bounding_box(actual, expected):
    actual_x_min, actual_y_min, actual_x_max, actual_y_max = actual
    expected_x_min, expected_y_min, expected_x_max, expected_y_max = expected

    intersection = [
        max(actual_x_min, expected_x_min),
        max(actual_y_min, expected_y_min),
        min(actual_x_max, expected_x_max),
        min(actual_y_max, expected_y_max),
    ]

    actual_area = calculate_area_bounding_box(actual)
    expected_area = calculate_area_bounding_box(expected)
    intersection_area = calculate_area_bounding_box(intersection)

    return intersection_area / (actual_area + expected_area - intersection_area)


def read_frames(path_to_folder):
    extension = ".jpg"
    frames = []
    file_name = 1
    file_path = path_to_folder + str(file_name) + extension
    print(file_path)
    while os.path.exists(file_path):
        frames.append(cv2.imread(file_path))
        file_name += 1
        file_path = path_to_folder + str(file_name) + extension

    return frames


if __name__ == '__main__':

    # ground truth bounding boxes will be shown with green
    ground_truth_bb_c = (0, 255, 0)
    # output bounding boxes will be shown with blue
    output_bb_c = (0, 0, 255)

    root_path = "dataset/TrainingSet/localisationFrames/"

    folder_names = [
        "Category I and II/",
        "Category III/",
        "Category IV/"
    ]

    ground_truth_file_name = "AABBGroundTruth.csv"

    for folder_name in folder_names:

        # import frames and ground truth
        path = root_path + folder_name
        frames = read_frames(path)
        number_of_plates = len(frames)

        ground_truth_path = path + ground_truth_file_name
        ground_truth_file = open(ground_truth_path)
        csvreader = csv.reader(ground_truth_file)
        # ignore header
        next(csvreader)

        # evaluation part
        scores = []
        not_recognised = 0
        for frame_num in range(len(frames)):

            frame = frames[frame_num]
            Localization.plate_detection(frame)

            expected = [eval(i) for i in next(csvreader)[1:5]]
            actual = Localization.localize_plates(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(frame, (expected[0], expected[1]), (expected[2], expected[3]), ground_truth_bb_c, 3)

            max_score = 0

            for aabb in actual:
                # Draw a rectangle on top of the original frame
                # indicating where our localization thinks the plate is.
                cv2.rectangle(frame, (aabb[0], aabb[1]), (aabb[2], aabb[3]), output_bb_c, 3)

                max_score = max(max_score, evaluate_bounding_box(aabb, expected))

            if max_score < 0.7:
                not_recognised += 1

            scores.append(max_score)

            if folder_name == "Category III/":
                expected1 = []
                expected1 = [eval(i) for i in next(csvreader)[1:5]]
                cv2.rectangle(frame, (expected1[0], expected1[1]), (expected1[2], expected1[3]), (0, 255, 0), 5)

                for aabb in actual:
                    max_score = max(max_score, evaluate_bounding_box(aabb, expected1))
                if max_score < 0.7:
                    not_recognised += 1
                scores.append(max_score)

            Commons.plotImage(frame, str(frame_num) + " " + str(max_score))

        average_score_of_category = np.average(scores)

        if folder_name == "Category III/":
            number_of_plates *= 2

        print(folder_name)
        print("Number of unrecognised plates: " + str(not_recognised) + " out of " + str(number_of_plates))
        print("Score: " + str(average_score_of_category))
        print()
