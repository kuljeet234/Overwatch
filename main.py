import cv2
import numpy as np
import face_recognition
import os

# YOLO setup
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes = ["Weapon"]

# Face recognition setup
known_face_encodings = []
known_face_names = []

# Load known faces
folder_path = "image"
for person_name in os.listdir(folder_path):
    person_directory = os.path.join(folder_path, person_name)
    if not os.path.isdir(person_directory):
        continue

    person_encodings = []
    for filename in os.listdir(person_directory):
        image_path = os.path.join(person_directory, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        person_encodings.append(encoding)

    known_face_encodings.extend(person_encodings)
    known_face_names.extend([person_name] * len(person_encodings))

# Video capture setup
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame from the video source.")
        break

    height, width, channels = frame.shape
    headcount = 0

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 corresponds to "Weapon"
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, "Weapon", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    # Face recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    headcount = len(face_locations)  # Update headcount based on the number of faces detected

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Red color for unknown faces

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Green color for known faces

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with detected objects and faces
    cv2.putText(frame, f"Headcount: {headcount}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
