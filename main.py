import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk
import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
from io import BytesIO

from databaseHandler import DatabaseHandler
from sort import *
import numpy as np
from datetime import datetime
from tkinter import filedialog
from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd


class ModelInitializer:
    def __init__(self):
        self.model = YOLO('NEW_openvino_model')
        self.classname = self.model.names[0]
        self.tracker = Sort(max_age=20, min_hits=3)


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main")
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")
        self.resizable(False, False)
        self.db_handler = DatabaseHandler(host="localhost", user="root", password="2003", database="warehouse")
        self.model_initializer = model_initializer
        self.initialize_video()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.db_handler.close_connection()
        self.destroy()

    def initialize_video(self):
        self.model = self.model_initializer.model
        self.classname = self.model_initializer.classname
        self.tracker = self.model_initializer.tracker
        self.line = [0, 800, 2560, 800]
        self.counter = []
        self.counterin = []
        self.counterout = []
        self.coordinates = []
        self.cap = None
        self.play_clicked = False
        self.create_widgets()

    def create_widgets(self):
        self.crossed_line_label = tk.Label(
            text=f'Crossed the line {len(self.counter)} times',
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        self.crossed_line_label.pack(side=tk.TOP, padx=5, pady=10)

        self.media_canvas = tk.Canvas(bg="black", width=900, height=500)
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

        self.pause_button = tk.Button(
            self.control_buttons_frame,
            text="Pause",
            font=("Arial", 12, "bold"),
            bg="#F44336",
            fg="white",
            command=self.pause_video,
        )
        self.pause_button.pack(side=tk.LEFT, pady=5)
        self.change_button_state("pause_button", tk.DISABLED)

        self.reset_button = tk.Button(
            self.control_buttons_frame,
            text="Reset",
            font=("Arial", 12, "bold"),
            bg="#FF5722",
            fg="white",
            command=self.reset_video_processing,
        )
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.change_button_state("reset_button", tk.DISABLED)

        self.view_results_button = tk.Button(
            self.control_buttons_frame,
            text="View table results",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            command=self.view_results,
        )
        self.view_results_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.view_charts_button = tk.Button(
            self.control_buttons_frame,
            text="View histogram results",
            font=("Arial", 12, "bold"),
            bg="#000080",
            fg="white",
            command=self.view_histogram,
        )
        self.view_charts_button.pack(side=tk.LEFT, padx=5, pady=5)

    def choose_video_file(self):
        video_file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp4 *.avi")])
        if video_file_path:
            self.cap = cv.VideoCapture(video_file_path)
            self.change_button_state("play_button", tk.NORMAL)
            self.change_button_state("reset_button", tk.NORMAL)
            self.change_button_state("view_charts_button", tk.DISABLED)
            self.change_button_state("view_results_button", tk.DISABLED)
            self.process_frame()

    def reset_video_processing(self):
        self.play_clicked = False
        self.change_button_state("pause_button", tk.DISABLED)
        self.change_button_state("view_results_button", tk.NORMAL)
        self.change_button_state("view_charts_button", tk.NORMAL)
        self.change_button_state("play_button", tk.DISABLED)
        self.counter = []
        self.crossed_line_label.config(text=f'Crossed the line {len(self.counter)} times')
        self.cap.release()
        self.media_canvas.delete("all")

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
            cvzone.putTextRect(img, self.classname, [x1 + 8, y1 - 12], scale=2, thickness=2)
            if self.coordinates and cy > self.coordinates[-1] and id == self.counter[-1] and self.counterout.count(
                    id) == 0:
                self.counterout.append(id)
                print("Виїзд")
            elif self.coordinates and cy < self.coordinates[-1] and id == self.counter[-1] and self.counterin.count(
                    id) == 0:
                self.counterin.append(id)
                print("Заїзд")
            if self.line[1] - 18 < cy < self.line[3] + 18:
                cv.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 0, 255), 10)
                if self.counter.count(id) == 0:
                    self.counter.append(id)
                    self.coordinates.append(cy)
                    image_encode = cv.imencode('.jpg', img)[1].tobytes()
                    self.db_handler.insert_data(datetime.now(), image_encode)
                    self.crossed_line_label.config(text=f'Crossed the line {len(self.counter)} times')

        frame = cv.resize(img, (800, 500))
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.media_canvas.imgtk = imgtk
        self.media_canvas.create_image(self.media_canvas.winfo_width() / 2, self.media_canvas.winfo_height() / 2,
                                       anchor=tk.CENTER, image=imgtk)

        if self.play_clicked:
            self.after(1, self.process_frame)

    def play_video(self):
        self.play_clicked = True
        self.change_button_state("play_button", tk.DISABLED)
        self.change_button_state("pause_button", tk.NORMAL)
        self.change_button_state("view_results_button", tk.DISABLED)
        self.change_button_state("view_charts_button", tk.DISABLED)
        if self.cap is not None:
            self.process_frame()

    def pause_video(self):
        self.play_clicked = False
        self.change_button_state("pause_button", tk.DISABLED)
        self.change_button_state("play_button", tk.NORMAL)

    def view_results(self):
        self.destroy()
        TableResultsApp().mainloop()

    def view_histogram(self):
        self.destroy()
        HistogramResultsApp().mainloop()

    def change_button_state(self, button_name, state):
        button = getattr(self, button_name)
        button.config(state=state)


class TableResultsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Table Results")
        self.resizable(False, False)
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")
        self.db_handler = DatabaseHandler(host="localhost", user="root", password="2003", database="warehouse")
        self.label = None
        self.create_widgets()

    def create_widgets(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('Treeview.Heading', background="green3")
        s.configure('Custom.Treeview', rowheight=25)

        self.title_label = tk.Label(self, text="Table results of goods crossed the line")
        self.title_label.pack(padx=10, pady=20, anchor="n")
        self.title_label.config(font=("Arial", 18), foreground="black")

        tree_frame = tk.Frame(self)
        tree_frame.pack(padx=(10, 0), side=tk.LEFT, anchor="n")

        self.tree = ttk.Treeview(tree_frame, columns=("Crossed Datetime",), show="headings", style="Custom.Treeview",
                                 height=23)
        self.tree.heading("Crossed Datetime", text="Crossed Datetime")
        self.tree.column("Crossed Datetime", anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")

        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(fill="both", expand=True)

        self.populate_treeview()

        self.placeholder_label = tk.Label(self, text="Click on table row to view image result")
        self.placeholder_label.pack(padx=10, anchor="n")
        self.placeholder_label.config(font=("Arial", 12), foreground="red")

        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

        self.control_buttons_frame = tk.Frame(bg="#f0f0f0")
        self.control_buttons_frame.pack(pady=5, side=tk.BOTTOM)

        self.refresh_button = tk.Button(
            self.control_buttons_frame,
            text="Refresh Table",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.update_treeview,
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.back_button = tk.Button(
            self.control_buttons_frame,
            text="Back to Main Page",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            command=self.back_to_video,
        )
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)

    def on_tree_click(self, event):
        item = self.tree.identify_row(event.y)

        if item:
            item = self.tree.selection()[0]
            value = self.tree.item(item, "values")[0]
            image_data = self.db_handler.read_image_by_datetime(value)
            image = Image.open(BytesIO(image_data))
            image = image.resize((750, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            if self.label:
                self.label.config(image=photo)
                self.label.image = photo
            else:
                self.placeholder_label.destroy()
                self.label = tk.Label(self, image=photo)
                self.label.image = photo
                self.label.pack(padx=10, anchor="n")

    def populate_treeview(self):
        data = self.db_handler.read_all_datetime_records()

        for i, datetime_record in enumerate(data):
            datetime_value = datetime_record[0]
            if i % 2 == 0:
                tag = "evenrow"
            else:
                tag = "oddrow"
            self.tree.insert("", "end", values=(datetime_value,), tags=(tag,))

        self.tree.tag_configure("evenrow", background="#f0f0f0")
        self.tree.tag_configure("oddrow", background="white")

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.populate_treeview()
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

    def back_to_video(self):
        self.destroy()
        MainApp().mainloop()


class HistogramResultsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Histogram Results")
        self.resizable(False, False)
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")
        self.db_handler = DatabaseHandler(host="localhost", user="root", password="2003", database="warehouse")
        self.create_widgets()

    def create_widgets(self):
        self.selection_frame = tk.Frame(self, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.selection_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=(20, 75))

        self.start_date_label = tk.Label(self.selection_frame, text="Select start date:",
                                         font=("Arial", 12, "bold"), bg="#ffffff")
        self.start_date_label.pack(pady=(30, 10), padx=20, anchor=tk.CENTER)

        self.start_datetime_picker = DateEntry(self.selection_frame, width=20, background='darkblue',
                                               foreground='white', borderwidth=2, state="readonly",
                                               date_pattern="dd.MM.yyyy")
        self.start_datetime_picker.pack(pady=10, padx=20, anchor=tk.CENTER)

        self.start_time_label = tk.Label(self.selection_frame, text="Select start time:",
                                         font=("Arial", 12, "bold"), bg="#ffffff")
        self.start_time_label.pack(pady=10, padx=20, anchor=tk.CENTER)

        self.start_time_frame = tk.Frame(self.selection_frame, bg="#ffffff")
        self.start_time_frame.pack(pady=(0, 10))

        self.start_hour_combobox = ttk.Combobox(self.start_time_frame, values=[f"{hour:02d}" for hour in range(24)],
                                                state="readonly", width=3)
        self.start_hour_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.start_hour_combobox.set("00")

        self.hour_label = tk.Label(self.start_time_frame, text="hr")
        self.hour_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.start_minute_combobox = ttk.Combobox(self.start_time_frame,
                                                  values=[f"{minute:02d}" for minute in range(60)],
                                                  state="readonly", width=3)
        self.start_minute_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.start_minute_combobox.set("00")

        self.minute_label = tk.Label(self.start_time_frame, text="min")
        self.minute_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.start_seconds_combobox = ttk.Combobox(self.start_time_frame,
                                                   values=[f"{second:02d}" for second in range(60)],
                                                   state="readonly", width=3)
        self.start_seconds_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.start_seconds_combobox.set("00")

        self.second_label = tk.Label(self.start_time_frame, text="sec")
        self.second_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.end_time_label = tk.Label(self.selection_frame, text="Select end date:",
                                       font=("Arial", 12, "bold"), bg="#ffffff")
        self.end_time_label.pack(pady=10, padx=20, anchor=tk.CENTER)

        self.end_datetime_picker = DateEntry(self.selection_frame, width=20, background='darkblue', foreground='white',
                                             borderwidth=2, state="readonly", date_pattern="dd.MM.yyyy")
        self.end_datetime_picker.pack(pady=10, padx=20, anchor=tk.CENTER)

        self.end_time_label = tk.Label(self.selection_frame, text="Select end time:",
                                       font=("Arial", 12, "bold"), bg="#ffffff")
        self.end_time_label.pack(pady=10, padx=20, anchor=tk.CENTER)

        self.end_time_frame = tk.Frame(self.selection_frame, bg="#ffffff")
        self.end_time_frame.pack(pady=(0, 10))

        self.end_hour_combobox = ttk.Combobox(self.end_time_frame, values=[f"{hour:02d}" for hour in range(24)],
                                              state="readonly", width=3)
        self.end_hour_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.end_hour_combobox.set("00")

        self.hour_label = tk.Label(self.end_time_frame, text="hr")
        self.hour_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.end_minute_combobox = ttk.Combobox(self.end_time_frame, values=[f"{minute:02d}" for minute in range(60)],
                                                state="readonly", width=3)
        self.end_minute_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.end_minute_combobox.set("00")

        self.minute_label = tk.Label(self.end_time_frame, text="min")
        self.minute_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.end_seconds_combobox = ttk.Combobox(self.end_time_frame, values=[f"{second:02d}" for second in range(60)],
                                                 state="readonly", width=3)
        self.end_seconds_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.end_seconds_combobox.set("00")

        self.second_label = tk.Label(self.end_time_frame, text="sec")
        self.second_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.selection_label = tk.Label(self.selection_frame, text="View results by:", font=("Arial", 12, "bold"),
                                        bg="#ffffff")
        self.selection_label.pack(pady=(20, 10))

        self.radio_button_container = tk.Frame(self.selection_frame, bg="#ffffff")
        self.radio_button_container.pack(pady=(0, 10))

        self.histogram_options = ["Years", "Months", "Days", "Hours"]
        self.selected_option = tk.StringVar(value=self.histogram_options[0])

        style = ttk.Style()
        style.configure("Custom.TRadiobutton", background="#ffffff")

        for option in self.histogram_options:
            ttk.Radiobutton(self.radio_button_container, text=option, value=option, variable=self.selected_option,
                            style="Custom.TRadiobutton").pack(
                padx=10, anchor=tk.W)

        self.build_histogram_button = tk.Button(
            self.selection_frame,
            text="Build Histogram",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.build_histogram
        )
        self.build_histogram_button.pack(pady=(20, 30), padx=10)

        self.back_button = tk.Button(
            text="Back to Main Page",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            command=self.back_to_video,
        )
        self.back_button.pack(side=tk.BOTTOM, pady=(0, 20))

        self.histogram_canvas = tk.Canvas(self, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.histogram_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=20)

    def build_histogram(self):
        selected_option = self.selected_option.get()
        start_date = self.start_datetime_picker.get_date()
        start_time = f"{self.start_hour_combobox.get()}:{self.start_minute_combobox.get()}:{self.start_seconds_combobox.get()}"
        end_date = self.end_datetime_picker.get_date()
        end_time = f"{self.end_hour_combobox.get()}:{self.end_minute_combobox.get()}:{self.end_seconds_combobox.get()}"

        start_datetime = pd.to_datetime(f"{start_date} {start_time}")
        end_datetime = pd.to_datetime(f"{end_date} {end_time}")

        if start_datetime >= end_datetime:
            messagebox.showerror("Error", "Start datetime must be earlier than end datetime.")
            return

        self.histogram_canvas.delete("all")

        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)

        data = self.fetch_data(start_datetime, end_datetime, selected_option)

        self.fig, ax = plt.subplots()
        n, bins, patches = ax.hist(data, color='skyblue', edgecolor='black', bins=20)
        ax.set_xlabel(selected_option)
        ax.set_ylabel('Count')
        ax.set_title('Histogram results of goods crossed the line\nTime interval: from %s to %s' % (
            start_datetime, end_datetime))
        ax.grid(True)

        ax.set_xticks(np.unique(data))

        for rect in patches:
            height = rect.get_height()
            ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.histogram_canvas)
        self.canvas.draw()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def fetch_data(self, start_datetime, end_datetime, selected_option):
        datetime_records = self.db_handler.read_datetime_records_in_range(start_datetime, end_datetime)

        df = pd.DataFrame(datetime_records, columns=['datetime'])

        if selected_option == "Years":
            data = df['datetime'].dt.year
        elif selected_option == "Months":
            data = df['datetime'].dt.month
        elif selected_option == "Days":
            data = df['datetime'].dt.day
        elif selected_option == "Hours":
            data = df['datetime'].dt.hour

        return data

    def back_to_video(self):
        self.destroy()
        MainApp().mainloop()


if __name__ == "__main__":
    model_initializer = ModelInitializer()
    app = MainApp()
    app.mainloop()
