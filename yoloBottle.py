import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np

cap = cv.VideoCapture('Test video.avi')
model = YOLO('NEW.pt')
# model = YOLO('NEW_openvino_model')

tracker = Sort(max_age=20, min_hits=3)
line = [130, 800, 2489, 800]
counterin = []

# print(model.names[0])
# with open('classes.txt', 'r') as f:
#     classnames = f.read().splitlines()

classname = model.names[0]
while 1:
    ret, img = cap.read()

    if not ret:
        cap = cv.VideoCapture('Test video.avi')
        continue
    detections = np.empty((0, 5))

    results = model(img, stream=True)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            conf = math.ceil(confidence * 100)
            if conf >= 50:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                current_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detections))

    tracker_result = tracker.update(detections)
    cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 5)

    for track_result in tracker_result:
        x1, y1, x2, y2, id = track_result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cvzone.cornerRect(img, [x1, y1, w, h], rt=5)
        cvzone.putTextRect(img, classname, [x1 + 8, y1 - 12],
                           scale=2, thickness=2)

        if line[1] - 18 < cy < line[3] + 18:
            cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
            if counterin.count(id) == 0:
                counterin.append(id)

    cvzone.putTextRect(img, f'Crossed the line {len(counterin)} times', [500, 34], thickness=4, scale=2.3, border=2)
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.imshow('frame', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
