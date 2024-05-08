import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
from io import BytesIO

from database import DatabaseHandler
from sort import *
import numpy as np
from datetime import datetime
from tkinter import filedialog


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main")
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")
        self.resizable(False, False)
        self.db_handler = DatabaseHandler(host="localhost", user="root", password="2003", database="warehouse")
        self.initialize_video()

    def initialize_video(self):
        self.model = YOLO('NEW_openvino_model')
        self.tracker = Sort(max_age=20, min_hits=3)
        self.line = [0, 800, 2560, 800]
        self.counterin = []
        self.desired_width = 900
        self.desired_height = 500
        self.cap = None
        self.create_widgets()

    def create_widgets(self):
        self.crossed_line_label = tk.Label(
            text=f'Crossed the line {len(self.counterin)} times',
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        self.crossed_line_label.pack(side=tk.TOP, padx=5, pady=10)

        self.media_canvas = tk.Canvas(bg="black", width=self.desired_width, height=self.desired_height)
        self.media_canvas.pack(pady=(0, 10), expand=False)

        self.select_file_button = tk.Button(
            text="Select File",
            font=("Arial", 12, "bold"),
            command=self.choose_video_file,
        )
        self.select_file_button.pack(pady=5)

        self.control_buttons_frame = tk.Frame(bg="#f0f0f0")
        self.control_buttons_frame.pack(pady=5)

        self.play_button = tk.Button(
            self.control_buttons_frame,
            text="Play",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.play_video,
        )
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.change_button_state("play_button", tk.DISABLED)


    def choose_video_file(self):
        video_file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp4 *.avi")])
        if video_file_path:
            self.cap = cv.VideoCapture(video_file_path)
            self.change_button_state("play_button", tk.NORMAL)
            self.process_frame()

    def process_frame(self):
        ret, img = self.cap.read()
        if not ret:
            self.cap.release()
            return

        detections = np.empty((0, 5))
        results = self.model(img, stream=True)
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

        tracker_result = self.tracker.update(detections)

        cv.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 255, 255), 5)
        for track_result in tracker_result:
            x1, y1, x2, y2, id = track_result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            cvzone.cornerRect(img, [x1, y1, w, h], rt=5)
            cvzone.putTextRect(img, self.model.names[0], [x1 + 8, y1 - 12], scale=2, thickness=2)
            if self.line[1] - 18 < cy < self.line[3] + 18:
                cv.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 0, 255), 10)
                if self.counterin.count(id) == 0:
                    self.counterin.append(id)
                    # image_encode = cv.imencode('.jpg', img)[1].tobytes()
                    # self.db_handler.insert_data(datetime.now(), image_encode)
                    self.crossed_line_label.config(text=f'Crossed the line {len(self.counterin)} times')

        frame = cv.resize(img, (800, 500))
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.media_canvas.imgtk = imgtk
        self.media_canvas.create_image(self.media_canvas.winfo_width() / 2, self.media_canvas.winfo_height() / 2,
                                       anchor=tk.CENTER, image=imgtk)

        self.after(25, self.process_frame)

    def play_video(self):
        self.change_button_state("play_button", tk.DISABLED)
        if self.cap is not None:
            self.process_frame()


    def change_button_state(self, button_name, state):
        button = getattr(self, button_name)
        button.config(state=state)



if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
