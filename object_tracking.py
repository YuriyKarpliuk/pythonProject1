import math

import cv2
from ultralytics import YOLO
import time

# model = YOLO("best.pt")  # build a new model from scratch

# classNames = ['qr-code']
myColor = (0, 0, 255)
cap = cv2.VideoCapture("Test video.avi")

while True:
    # start = time.time()
    success, img = cap.read()
    # end = time.time()
    # fps = math.ceil(1 / (end - start))

    # results = model.track(img, persist=True)
    #
    # frame_ = results[0].plot()

    # cv2.putText(img, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, myColor, 2)
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode
#
# cap = cv2.VideoCapture("Test video.avi")

# def find_center(pt1, pt2):
#     center_x = (pt1[0] + pt2[0]) / 2
#     center_y = (pt1[1] + pt2[1]) / 2
#     return center_x, center_y
#
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# # Define line coordinates
# line_pt1 = (130, 1200)
# line_pt2 = (2489, 1260)
#
# # Initialize counter
# counter = 0
#
# center_of_line = find_center(line_pt1, line_pt2)
# print("Center of the line:", center_of_line)
# print("Center of the line y :", center_of_line[1])
# while True:
#     success, img = cap.read()
#
#     if not success:
#         break
#
#     for barcode in decode(img):
#         decoded_data = barcode.data.decode("utf-8")
#         rect_pts = barcode.rect
#
#         if decoded_data:
#             pts = np.array([barcode.polygon], np.int32)
#             cv2.polylines(img, [pts], True, (0, 255, 0), 3)
#         # pts = np.array([barcode.polygon], np.int32)
#         # pts = pts.reshape((-1, 1, 2))
#         # cv2.polylines(img, [pts], True, (0, 255, 0), 5)
#         # center = np.mean(pts, axis=0, dtype=np.int32)
#         #
#         # print(center[0][1])
#         # if center[0][1] > center_of_line[1]:
#         #     counter += 1
#         #     print(counter)
#
#     # Draw line
#     # cv2.line(img=img, pt1=line_pt1, pt2=line_pt2, color=(255, 0, 0), thickness=5)
#     #
#     # # Display counter at the edge of the screen
#     # cv2.putText(img, f'Count: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
#     cv2.imshow('Video', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture("Test video.avi")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, img = cap.read()

    if not success:
        break

    for barcode in decode(img):
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 5)
        center = np.mean(pts, axis=0, dtype=np.int32)

        # Виведення координат центру
        print("Центр QR-коду:", center)
    cv2.line(img=img, pt1=(130, 1300), pt2=(2489, 1395), color=(255, 0, 0), thickness=5)

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# for r in results:
#     boxes = r.boxes
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         # w, h = x2 - x1, y2 - y1
#         # cvzone.cornerRect(img, (x1, y1, w, h))
#         cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

# conf = math.ceil((box.conf[0] * 100)) / 100
# cls = int(box.cls[0])
# cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
#                    (max(0, x1), max(35, y1)), scale=1, thickness=1,
#                    colorB=myColor, colorT=(255, 255, 255))
#     cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
#
# cv2.imshow("Frame", img)
# cv2.waitKey(1)
# , frame = cap.read()


# import cv2
#
# cpt = 0
# maxFrames = 110  # if you want 5 frames only.
#
# count = 0
# cap = cv2.VideoCapture('Test video.avi')
# while cpt < maxFrames:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     count += 1
#     if count % 3 != 0:
#         continue
#     frame = cv2.resize(frame, (2560, 1920))
#     # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
#     cv2.imshow("Frame", frame)  # show image in window
#     cv2.imwrite(r"D:\MyProjects\frames\pack-b_%d.jpg" % cpt, frame)
#     cpt += 1
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()
