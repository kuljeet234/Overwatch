# Overwatch

CCTV-style monitor that combines a custom-trained YOLOv3 weapon
detector with face recognition. Per frame it draws boxes around
detected weapons, labels known/unknown faces, and prints a live
headcount.

## Setup

```
pip install -r requirements.txt
./bootstrap.sh
```

`bootstrap.sh` creates the directory layout and prints where to
place the trained weights file. The expected layout is:

```
yolov3_training_2000.weights   # custom-trained YOLO weights (not committed)
yolov3_testing.cfg             # network definition (committed)
image/
  alice/
    photo1.jpg
    photo2.jpg
  bob/
    photo1.jpg
```

## Run

```
python main.py                # default webcam
python main.py --camera 1     # second webcam
python main.py --video clip.mp4
python main.py --threshold 0.6
```

Press `q` to quit.
