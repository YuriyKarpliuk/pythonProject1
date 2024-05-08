import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from database import DatabaseHandler
import numpy as np
from tkcalendar import DateEntry


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
        self.selection_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.start_date_label = tk.Label(self.selection_frame, text="Select start date:",
                                         font=("Arial", 12, "bold"), bg="#ffffff")
        self.start_date_label.pack(pady=(20, 5), padx=20, anchor=tk.CENTER)

        self.start_datetime_picker = DateEntry(self.selection_frame, width=20, background='darkblue',
                                               foreground='white', borderwidth=2, state="readonly",
                                               date_pattern="dd.MM.yyyy")
        self.start_datetime_picker.pack(pady=(0, 10), padx=20, anchor=tk.CENTER)

        self.start_time_label = tk.Label(self.selection_frame, text="Select start time:",
                                         font=("Arial", 12, "bold"), bg="#ffffff")
        self.start_time_label.pack(pady=(0, 5), padx=20, anchor=tk.CENTER)

        self.start_time_frame = tk.Frame(self.selection_frame, bg="#ffffff")
        self.start_time_frame.pack(pady=(0, 10))

        self.hour_combobox = ttk.Combobox(self.start_time_frame, values=[f"{hour:02d}" for hour in range(24)],
                                      state="readonly", width=3)
        self.hour_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.hour_combobox.set("00")

        self.hour_label = tk.Label(self.start_time_frame, text="hr")
        self.hour_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.minute_combobox = ttk.Combobox(self.start_time_frame, values=[f"{minute:02d}" for minute in range(60)],
                                            state="readonly", width=3)
        self.minute_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.minute_combobox.set("00")

        self.minute_label = tk.Label(self.start_time_frame, text="min")
        self.minute_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.second_combobox = ttk.Combobox(self.start_time_frame, values=[f"{second:02d}" for second in range(60)],
                                            state="readonly", width=3)
        self.second_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.second_combobox.set("00")

        self.second_label = tk.Label(self.start_time_frame, text="sec")
        self.second_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.end_time_label = tk.Label(self.selection_frame, text="Select end date:",
                                       font=("Arial", 12, "bold"), bg="#ffffff")
        self.end_time_label.pack(pady=(0, 5), padx=20, anchor=tk.CENTER)

        self.end_datetime_picker = DateEntry(self.selection_frame, width=20, background='darkblue', foreground='white',
                                             borderwidth=2, state="readonly", date_pattern="dd.MM.yyyy")
        self.end_datetime_picker.pack(pady=(0, 10), padx=20, anchor=tk.CENTER)

        self.end_time_label = tk.Label(self.selection_frame, text="Select end time:",
                                         font=("Arial", 12, "bold"), bg="#ffffff")
        self.end_time_label.pack(pady=(0, 5), padx=20, anchor=tk.CENTER)

        self.end_time_frame = tk.Frame(self.selection_frame, bg="#ffffff")
        self.end_time_frame.pack(pady=(0, 10))

        self.hour_combobox = ttk.Combobox(self.end_time_frame, values=[f"{hour:02d}" for hour in range(24)],
                                          state="readonly", width=3)
        self.hour_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.hour_combobox.set("00")

        self.hour_label = tk.Label(self.end_time_frame, text="hr")
        self.hour_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.minute_combobox = ttk.Combobox(self.end_time_frame, values=[f"{minute:02d}" for minute in range(60)],
                                            state="readonly", width=3)
        self.minute_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.minute_combobox.set("00")

        self.minute_label = tk.Label(self.end_time_frame, text="min")
        self.minute_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.second_combobox = ttk.Combobox(self.end_time_frame, values=[f"{second:02d}" for second in range(60)],
                                            state="readonly", width=3)
        self.second_combobox.pack(pady=5, padx=5, side=tk.LEFT)
        self.second_combobox.set("00")

        self.second_label = tk.Label(self.end_time_frame, text="sec")
        self.second_label.pack(pady=3, padx=3, side=tk.LEFT)

        self.selection_label = tk.Label(self.selection_frame, text="View results by:", font=("Arial", 12, "bold"),
                                        bg="#ffffff")
        self.selection_label.pack(pady=(10, 5))

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
        self.build_histogram_button.pack(side=tk.BOTTOM, pady=(20, 10), padx=10)

        self.back_button = tk.Button(
            text="Back to Main Page",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            command=self.back_to_video,
        )
        self.back_button.pack(side=tk.BOTTOM, pady=(0, 20))

        self.histogram_canvas = tk.Canvas(self, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.histogram_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def build_histogram(self):
        selected_option = self.selected_option.get()

        self.histogram_canvas.delete("all")

        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)

        data = self.fetch_data(selected_option)

        # if selected_option == "Years":
        #     bins = np.arange(data.min(), data.max() + 1)
        #     xticks = range(data.min(), data.max() + 1)
        # elif selected_option == "Months":
        #     bins = np.arange(0.5, 13, 1)
        #     xticks = range(1, 13)
        # elif selected_option == "Days":
        #     bins = np.arange(0.5, 32, 1)
        #     xticks = range(1, 32)
        # elif selected_option == "Hours":
        #     bins = np.arange(-0.5, 24, 1)
        #     xticks = range(0, 24)

        self.fig, ax = plt.subplots()
        ax.hist(data, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_option)
        ax.set_ylabel('Count')
        ax.set_title('Histogram')
        ax.grid(True)

        # ax.set_xticks(xticks)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.histogram_canvas)
        self.canvas.draw()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def fetch_data(self, selected_option):
        datetime_records = self.db_handler.read_all_datetime_records()

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


if __name__ == "__main__":
    app = HistogramResultsApp()
    app.mainloop()
