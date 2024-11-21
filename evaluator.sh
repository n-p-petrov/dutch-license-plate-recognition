#!/bin/bash

T=$(find ./dataset -type f -name "dummytestvideo.avi")

python main.py --file_path $T --output_path ./Output.csv

F=$(find . -type f -name "Output.csv")

G=$(find ./dataset -type f -name "dummyGroundTruthTest.csv")

python evaluation.py --file_path $F  --ground_truth_path $G

sleep 100s
