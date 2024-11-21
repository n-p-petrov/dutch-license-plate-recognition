import os
import cv2
import numpy as np
import pandas as pd

import Recognize


def evaluate_plates():
    plates_dir = 'dataset/ValidationSet/recognitionPlates/Category I and II/'
    ground_truth_path = 'dataset/ValidationSet/recognitionPlates/Category I and II/PlateGroundTruth.csv'
    ground_truth = pd.read_csv(ground_truth_path)
    plates = ground_truth['License plate']

    count = 0
    total = 0
    for fname in os.listdir(plates_dir):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            boo = False
            num = fname.split(".")[0]
            plate_image = cv2.imread(plates_dir + fname)
            letters = Recognize.segment_and_recognize(plate_image)
            if letters == plates[int(num) - 1]:
                count += 1
                boo = True
            total += 1
    percentage = count/total * 100
    print("Percentage: " + str(percentage))
    #Percentage:  84.375% | T = 0.08 | C = 15 | R = 0.9500000000000004
def hyperparameter_tuning():
    for threshold in np.arange(0.08, 0.20, 0.01):
        for mean_c in np.arange(1, 20, 1):
            for xor_ratio in np.arange(0.5, 1, 0.05):
                evaluate_plates(threshold, mean_c, xor_ratio)