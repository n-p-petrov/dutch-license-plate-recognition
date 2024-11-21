import argparse
import csv
import os
import CaptureFrame_Process
import cv2
import Commons
import Recognize
from Localization import plate_detection


def recognition_temp_test():
    im_path = "dataset/TrainingSet/recognitionPlates/Category I and II/1.jpg"
    image = cv2.imread(im_path)
    Recognize.segment_and_recognize(image)


def localization_temp_test():
    cap = cv2.VideoCapture("dataset/trainingsvideo.avi")

    # Choose a frame to work on
    frameN = 1700

    for i in range(0, frameN):

        # Read the video frame by frame
        ret, frame = cap.read()
        # if we have no more frames end the loop
        if not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    Commons.plotImage(frame, "frame")
    plate_detection(frame)


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/trainingsvideo.avi')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--sample_frequency', type=int, default=12)
    args = parser.parse_args()
    return args


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
    args = get_args()
    if args.output_path is None:
        output_path = os.getcwd()
    else:
        output_path = args.output_path
    file_path = args.file_path
    sample_frequency = args.sample_frequency

    # file_path = "./dataset/TrainingSet/TrainingVideo.mp4"
    # output_path = "./Output.csv"
    # sample_frequency = 12

    results = CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency)

    file = open(output_path, 'w')
    writer = csv.writer(file)
    data = ["License plate", "Frame no.", "Timestamp(seconds)"]
    writer.writerow(data)
    for result in results:
        print(result)
        data = [result[0], result[1], result[2]]
        writer.writerow(data)
    file.close()
