import cv2
import mediapipe as mp
import dlib
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load Dlib face detector & predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load YOLO model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layers = net.getLayerNames()
out_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

# Face Detection (Using Mediapipe)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Head Pose Estimation
def head_pose_estimation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
        cv2.circle(frame, tuple(nose_tip), 3, (255, 0, 0), -1)
    return frame

# Object Detection (YOLO)
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(out_layers)
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame

# Start Video Capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_face(frame)
    frame = head_pose_estimation(frame)
    frame = detect_objects(frame)
    cv2.imshow("Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
