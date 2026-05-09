"""CCTV-style monitor: YOLO weapon detection + face recognition + headcount.

Run:
    python main.py            # opens default webcam (0)
    python main.py --camera 1 # different webcam
    python main.py --video clip.mp4
"""
import argparse
import os
import sys


WEIGHTS_PATH = "yolov3_training_2000.weights"
CONFIG_PATH = "yolov3_testing.cfg"
FACES_DIR = "image"
WEAPON_CLASS_NAMES = ["Weapon"]
# Default tolerance from face_recognition; lower is stricter.
FACE_MATCH_TOLERANCE = 0.6


def _import_runtime_deps():
    """Import the heavy CV/ML libs lazily so `python main.py --help` still
    works on a host that hasn't pip-installed them yet."""
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        import face_recognition  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"\nMissing dependency: {exc.name}.\n"
            "Run: pip install -r requirements.txt\n"
            "(face-recognition pulls dlib which requires a C++ toolchain — see requirements.txt notes.)\n"
        )


def load_yolo(weights=WEIGHTS_PATH, config=CONFIG_PATH):
    import cv2
    if not os.path.exists(weights):
        sys.exit(
            f"\nMissing {weights}.\n"
            "This is a custom-trained YOLOv3 weight file (not committed to the repo).\n"
            "Place yours at this path, or train one with the Darknet pipeline at\n"
            "https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects\n"
        )
    if not os.path.exists(config):
        sys.exit(f"\nMissing {config} (the YOLO network config). It should be in the repo.\n")
    net = cv2.dnn.readNet(weights, config)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_known_faces(folder=FACES_DIR):
    """Load one or more reference images per person from {folder}/{name}/*.jpg.
    Files where face_recognition can't find a face are skipped with a warning
    instead of crashing the whole startup. face_recognition.load_image_file
    returns RGB, which is what dlib's encoding expects.
    """
    import face_recognition
    encodings = []
    names = []
    if not os.path.isdir(folder):
        print(f"warning: {folder}/ does not exist — no known faces will be recognised.")
        return encodings, names

    for person_name in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            image_path = os.path.join(person_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)
            except Exception as e:
                print(f"  skipping {image_path}: {e}")
                continue
            faces = face_recognition.face_encodings(image)
            if not faces:
                print(f"  skipping {image_path}: no face detected")
                continue
            encodings.append(faces[0])
            names.append(person_name)
    print(f"Loaded {len(encodings)} face encodings for {len(set(names))} people.")
    return encodings, names


def open_capture(camera, video):
    import cv2
    if video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            sys.exit(f"Could not open video file: {video}")
    else:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            sys.exit(f"Could not open camera index {camera}.")
    return cap


def detect_weapons(net, frame, conf_threshold=0.5):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold and class_id == 0:
                cx, cy = int(detection[0] * w), int(detection[1] * h)
                bw, bh = int(detection[2] * w), int(detection[3] * h)
                boxes.append((cx - bw // 2, cy - bh // 2, bw, bh))
    return boxes


def best_match(known_encodings, face_encoding, tolerance=FACE_MATCH_TOLERANCE):
    """Pick the closest known encoding by face_distance, or None if all are
    further than `tolerance`. matches.index(True) was wrong because it
    picked the first encoding past the threshold, ignoring distance.
    """
    import face_recognition
    import numpy as np
    if not known_encodings:
        return None
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    idx = int(np.argmin(distances))
    if distances[idx] <= tolerance:
        return idx
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--video", help="Path to a video file (overrides --camera)")
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--faces-dir", default=FACES_DIR)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="YOLO confidence threshold (default: 0.5)")
    parser.add_argument("--match-tolerance", type=float, default=FACE_MATCH_TOLERANCE,
                        help="Face match tolerance, lower is stricter (default: 0.6)")
    args = parser.parse_args()

    _import_runtime_deps()
    import cv2
    import face_recognition

    net = load_yolo(args.weights, args.config)
    known_encodings, known_names = load_known_faces(args.faces_dir)
    cap = open_capture(args.camera, args.video)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed — exiting.")
            break

        for x, y, w, h in detect_weapons(net, frame, args.threshold):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Weapon", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # face_recognition is built on dlib, which expects RGB. cv2 frames
        # are BGR — convert before encoding so encodings match the
        # reference photos (which load_image_file already returns as RGB).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        headcount = len(face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            idx = best_match(known_encodings, face_encoding, args.match_tolerance)
            if idx is not None:
                name = known_names[idx]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
