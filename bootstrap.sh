#!/usr/bin/env bash
# Set up the directory layout main.py expects.
# Does NOT download the YOLO weights — those are a custom-trained
# artifact specific to your training run. Drop yours in at the path
# the script prints if it isn't already there.
set -euo pipefail

mkdir -p image
echo "Created image/ for known-face folders."
echo "Layout main.py expects:"
echo "  image/<person_name>/photo1.jpg"
echo "  image/<person_name>/photo2.jpg"
echo

WEIGHTS="yolov3_training_2000.weights"
if [ ! -f "$WEIGHTS" ]; then
  echo "Place your trained YOLOv3 weights at: $WEIGHTS"
  echo "If you don't have weights yet, train via:"
  echo "  https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects"
else
  echo "$WEIGHTS already in place."
fi
