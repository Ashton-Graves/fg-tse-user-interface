import tkinter as tk
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib
import pygame
matplotlib.use("tkAgg")
import json
import librosa
import tkinter.ttk as ttk
from pydub import AudioSegment
import subprocess
import os
import random

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

pygame.mixer.pre_init(44100, -16, 2, 512)  # Adjust buffer size if needed
pygame.mixer.init()

sd.default.samplerate = 44100 

class MainFrame(tk.Tk):
    def __init__(self):

        super(MainFrame, self).__init__()
        self.title("Target Speech Extraction User Interface")
        self.geometry('800x600')

        # Allow selection for clips to be used
        self.var = tk.StringVar(value="1-25")  # Default selection

        self.released = False
        self.new_time = 0
        self.counter = 1
        self.isPaused = False  # Initialize isPaused here
        

        self.enr_sr = 44100  # example sample rate for enrollment
        self.mix_sr = 44100  # example sample rate for mixture


        file_nums_array = np.linspace(0, 99, 100, dtype=int)
        strings = ["-10-0", "-10000--10", "0-10", "10-10000"]

        # Use np.tile to repeat the file_nums_array for each string, and np.repeat to repeat the strings accordingly
        dir_array = np.array([f"{s}/{str(num).zfill(4)}" for num in file_nums_array for s in strings])

        np.random.seed(43)

        np.random.shuffle(dir_array)
        print(dir_array)

        # Initialize an empty dictionary to hold your file arrays
        self.files_dict = {}

        # Number of selections (50 per set)
        selection_size = 25

        # Loop to create selections (from 1 to 11)
        for i in range(1, 11):
            start_index = (i - 1) * selection_size
            end_index = i * selection_size
            self.files_dict[f"files{i*25-24}-{i*25}"] = dir_array[start_index:end_index]

        self.setup_directory()
        self.prepare_audio()

        self.itr_log = []
        self.complete_log = []

        # Label to display the selected value
        self.itr_label = tk.Label(self, text="Example 1/25")
        self.itr_label.pack(pady=10)

        self.buttonframe1 = tk.Frame(self)
        self.buttonframe1.columnconfigure(0, weight=1)
        self.buttonframe1.columnconfigure(1, weight=1)

        self.rectangles = []


        self.addHighlightButton = tk.Button(self.buttonframe1, text="Highlight", command = self.highlight)
        self.addHighlightButton.grid(row=0, column=0)

        self.addUndoButton = tk.Button(self.buttonframe1, text="↺", command = self.undoHighlight)
        self.addUndoButton.grid(row=0, column=1)

        self.buttonframe1.pack()

        self.buttonframe2 = tk.Frame(self)
        self.buttonframe2.columnconfigure(0, weight=1)
        self.buttonframe2.columnconfigure(1, weight=1)

        pygame.mixer.music.set_volume(0.35)

        self.addPlayMix = tk.Button(self, text = "Play mixture", command=self.play_mix)
        self.addPlayMix.pack()

        self.addPlayEnr = tk.Button(self, text = "Play enrollment", command=self.play_enr)
        self.addPlayEnr.pack()

        self.addPlayButton = tk.Button(self.buttonframe2, text="▶", font=("Arial", 20), command=self.play)
        self.addPlayButton.grid(row=0, column=0)

        self.addPauseButton = tk.Button(self.buttonframe2, text="⏸", font=("Arial", 20), command=self.pause)
        self.addPauseButton.grid(row=0, column=1)

        self.buttonframe2.pack()

        self.draw_figure()
        
        self.playback_bar = self.ax3.axvspan(0, 0.001, color="gray")

        self.addRefineButton = tk.Button(self, text = "Refine", command=self.refine)
        self.addRefineButton.pack(side="right")

        # Create a frame for the buttons at the bottom
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side="bottom", fill="x")
        # Add radio buttons to the frame with grid layout

        self.addButton25 = tk.Radiobutton(bottom_frame, variable=self.var, text="1-25", value="1-25", command=self.mode)
        self.addButton25.grid(row=0, column=0)

        self.addButton50 = tk.Radiobutton(bottom_frame, variable=self.var, text="26-50", value="26-50", command=self.mode)
        self.addButton50.grid(row=0, column=1)
        
        self.addButton75 = tk.Radiobutton(bottom_frame, variable=self.var, text="51-75", value="51-75", command=self.mode)
        self.addButton75.grid(row=0, column=2)
        
        self.addButton100 = tk.Radiobutton(bottom_frame, variable=self.var, text="76-100", value="76-100", command=self.mode)
        self.addButton100.grid(row=0, column=3)
        
        self.addButton125 = tk.Radiobutton(bottom_frame, variable=self.var, text="101-125", value="101-125", command=self.mode)
        self.addButton125.grid(row=0, column=4)
        
        self.addButton150 = tk.Radiobutton(bottom_frame, variable=self.var, text="126-150", value="126-150", command=self.mode)
        self.addButton150.grid(row=1, column=0)
        
        self.addButton175 = tk.Radiobutton(bottom_frame, variable=self.var, text="151-175", value="151-175", command=self.mode)
        self.addButton175.grid(row=1, column=1)
        
        self.addButton200 = tk.Radiobutton(bottom_frame, variable=self.var, text="176-200", value="176-200", command=self.mode)
        self.addButton200.grid(row=1, column=2)
        
        self.addButton225 = tk.Radiobutton(bottom_frame, variable=self.var, text="201-225", value="201-225", command=self.mode)
        self.addButton225.grid(row=1, column=3)

        self.addButton250 = tk.Radiobutton(bottom_frame, variable=self.var, text="226-250", value="226-250", command=self.mode)
        self.addButton250.grid(row=1, column=4)
        


        # Label to display the selected value
        self.label = tk.Label(self, text="You selected: 1-25")
        self.label.pack(pady=10)


    def slide(self, x):
        self.new_time = self.slider.get()
        pygame.mixer.music.play(start=float(self.new_time))
        pygame.mixer.music.pause()
        self.playback_bar.remove()  # Remove the old bar
        # Update the playback bar
        self.playback_bar = self.ax3.axvspan(self.new_time, self.new_time + 0.001, color="gray")
        self.canvas.draw()
        
    def setup_directory(self):
        self.dir = f"./user_study_survey_data/"
        current_index = (self.counter % 25) - 1
        
        if current_index == -1:  # When counter is a multiple of 20, set index to 19
            current_index = 24

        print("current_index: ", current_index)
        
        if (self.var.get() == "1-25"):
            self.dir += self.files_dict["files1-25"][current_index]
        elif (self.var.get() == "26-50"):
            self.dir += self.files_dict["files26-50"][current_index]
        elif (self.var.get() == "51-75"):
            self.dir += self.files_dict["files51-75"][current_index]
        elif (self.var.get() == "76-100"):
            self.dir += self.files_dict["files76-100"][current_index]
        elif (self.var.get() == "101-125"):
            self.dir += self.files_dict["files101-125"][current_index]
        elif (self.var.get() == "126-150"):
            self.dir += self.files_dict["files126-150"][current_index]
        elif (self.var.get() == "151-175"):
            self.dir += self.files_dict["files151-175"][current_index]
        elif (self.var.get() == "176-200"):
            self.dir += self.files_dict["files176-200"][current_index]
        elif (self.var.get() == "201-225"):
            self.dir += self.files_dict["files201-225"][current_index]
        elif (self.var.get() == "226-250"):
            self.dir += self.files_dict["files226-250"][current_index]

    def mode(self):
        # Update the label text with the selected value
        self.label.config(text=f"You selected: {self.var.get()}")
        if (self.var.get() == "1-25"):
            self.counter = 1
        elif (self.var.get() == "26-50"):
            self.counter = 26
        elif (self.var.get() == "51-75"):
            self.counter = 51
        elif (self.var.get() == "76-100"):
            self.counter = 76
        elif (self.var.get() == "101-125"):
            self.counter = 101
        elif (self.var.get() == "126-150"):
            self.counter = 126
        elif (self.var.get() == "151-175"):
            self.counter = 151
        elif (self.var.get() == "176-200"):
            self.counter = 176
        elif (self.var.get() == "201-225"):
            self.counter = 201
        elif (self.var.get() == "226-250"):
            self.counter = 226

        if (self.var.get() == "1-25"):
            self.itr_label.config(text=f"Example {self.counter}/25")

        elif (self.var.get() == "26-50"):
            self.itr_label.config(text=f"Example {self.counter}/50")

        elif (self.var.get() == "51-75"):
            self.itr_label.config(text=f"Example {self.counter}/75")

        elif ((self.var.get() == "76-100")):
            self.itr_label.config(text=f"Example {self.counter}/100")
        
        elif ((self.var.get() == "101-125")):
            self.itr_label.config(text=f"Example {self.counter}/125")
        
        elif ((self.var.get() == "126-150")):
            self.itr_label.config(text=f"Example {self.counter}/150")
        
        elif ((self.var.get() == "151-175")):
            self.itr_label.config(text=f"Example {self.counter}/175")
        
        elif ((self.var.get() == "176-200")):
            self.itr_label.config(text=f"Example {self.counter}/200")
        
        elif ((self.var.get() == "201-225")):
            self.itr_label.config(text=f"Example {self.counter}/225")
        
        elif ((self.var.get() == "225-250")):
            self.itr_label.config(text=f"Example {self.counter}/250")
        

        self.setup_directory()
        self.prepare_audio()

        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.draw_waveform()
        self.canvas.draw()

        self.complete_log = []
        self.itr_log = []

    def prepare_audio(self):
        self.file = self.dir + "/tse_output.wav"
        audio = AudioSegment.from_wav(self.file)
        output_file = str(self.file) + ".ogg"
        audio.export(output_file, format="ogg", codec="libvorbis")
        pygame.mixer.music.load(output_file)
        print(self.file)

    def draw_waveform(self):
        self.waveform, self.sr = sf.read(self.file)
        time = np.linspace(0, len(self.waveform) / self.sr, num=len(self.waveform))
        # Choose a downsampling factor
        downsample_factor = 50

        self.mixture = self.dir + "/mixture.wav"
        self.mix_waveform, self.mix_sr = sf.read(self.mixture)
        mix_time = np.linspace(0, len(self.mix_waveform) / self.mix_sr, num=len(self.mix_waveform))

        mix_downsampled_time = mix_time[::downsample_factor]
        mix_downsampled_waveform = self.mix_waveform[::downsample_factor]
        mix_downsampled_waveform = mix_downsampled_waveform / np.max(np.abs(mix_downsampled_waveform))
        self.ax.plot(mix_downsampled_time, mix_downsampled_waveform, color='red')
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title("mixture")

        # Downsample the time and waveform
        downsampled_time = time[::downsample_factor]
        downsampled_waveform = self.waveform[::downsample_factor]
        downsampled_waveform = downsampled_waveform / np.max(np.abs(downsampled_waveform))
        self.ax3.plot(downsampled_time, downsampled_waveform, color='red')
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax3.set_title("output")

        self.enrollment = self.dir + "/enrollment.wav"
        self.enr_waveform, self.enr_sr = sf.read(self.enrollment)
        enr_time = np.linspace(0, len(self.enr_waveform) / self.enr_sr, num=len(self.enr_waveform))

        enr_downsampled_time = enr_time[::downsample_factor]
        enr_downsampled_waveform = self.enr_waveform[::downsample_factor]
        enr_downsampled_waveform = enr_downsampled_waveform / np.max(np.abs(enr_downsampled_waveform))
        self.ax2.plot(enr_downsampled_time, enr_downsampled_waveform, color='red')
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_title("enrollment")

        # Compute global y-axis limits
        y_min = min(np.min(mix_downsampled_waveform), np.min(downsampled_waveform), np.min(enr_downsampled_waveform))
        y_max = max(np.max(mix_downsampled_waveform), np.max(downsampled_waveform), np.max(enr_downsampled_waveform))

        # Set the same y-axis limits for all subplots
        for axis in [self.ax, self.ax2, self.ax3]:
            axis.set_ylim(y_min, y_max)  # Ensures same y-scale across plots
            axis.relim()
            axis.autoscale_view()
            axis.set_xlabel("time")


    def draw_figure(self):
        plt.style.use('dark_background') 

        # Create the figure and axis using the object-oriented API
        self.fig = Figure(figsize=(5, 3))
        self.fig.subplots_adjust(hspace=1.2)
        self.ax = self.fig.add_subplot(311)  # First subplot
        self.ax2 = self.fig.add_subplot(312)  # Second subplot
        self.ax3 = self.fig.add_subplot(313)  # Second subplot

        # Configure axis appearance
        self.ax.tick_params(
            axis='both', which='both',
            bottom=False, top=False, labelbottom=False,
            right=False, left=False, labelleft=False,
            labelcolor="white", colors="white"
        )

        self.ax2.tick_params(
            axis='both', which='both',
            bottom=False, top=False, labelbottom=False,
            right=False, left=False, labelleft=False,
            labelcolor="white", colors="white"
        )

        self.ax3.tick_params(
            axis='both', which='both',
            bottom=False, top=False, labelbottom=False,
            right=False, left=False, labelleft=False,
            labelcolor="white", colors="white"
        )

        self.draw_waveform()

        # Embed the figure in the Tkinter frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(anchor=tk.CENTER, fill=tk.BOTH, expand=True)  # Pack canvas to fill available space

        # Draw the canvas
        self.canvas.draw()

        # Pack the slider under the canvas
        self.slider = ttk.Scale(self, from_=0, to_=len(self.waveform) / self.sr, 
                                orient="horizontal", value=0, command=self.slide)
        self.slider.pack(anchor=tk.CENTER, fill=tk.X, side=tk.TOP, padx=(0.5*self.winfo_width(), 0.5*self.winfo_width()))  # Automatically match canvas width


    def play_enr(self):
        sd.stop()
        data, samplerate = sf.read(self.enrollment)
        data = data * 0.3 # reduce volume
        sd.play(data, self.enr_sr)
        sd.wait()
        sd.stop()
        
    def play_mix(self):
        sd.stop()
        data, samplerate = sf.read(self.mixture)
        data = data * 0.3 # reduce volume
        sd.play(data, self.mix_sr)
        sd.wait()
        sd.stop()

    def play(self):
        sd.stop()
        if self.isPaused:
            pygame.mixer.music.unpause()
            self.isPaused = False
            self.update_playback_bar()
        elif not pygame.mixer.music.get_busy():
            self.new_time = 0
            self.slider_pos = self.slider.get()
            pygame.mixer.music.play(start=float(self.slider_pos))
            self.update_playback_bar()
            self.slider.config(value=0)

        
    def pause(self):
        sd.stop()
        if (self.isPaused):
            pygame.mixer.music.unpause()
            self.isPaused = False
            self.update_playback_bar()
        else:
            pygame.mixer.music.pause()
            self.isPaused = True
            self.update_playback_bar()

        

    def highlight(self):
        # Connect the canvas to listen for mouse events
        self.press_event = self.canvas.mpl_connect("button_press_event", self.on_press)

    def undoHighlight(self):
        if (len(self.rectangles) > 0):
            patch = self.rectangles.pop()  # Remove the last rectangle
            patch.remove()  # Remove it from the canvas
            self.canvas.draw()  # Redraw the canvas to reflect the changes
            self.itr_log.pop()
    
    def on_press(self, event):
        # Start the selection when the mouse button is pressed
        if event.inaxes == self.ax3:
            self.start_x = event.xdata  # Record the starting x-coordinate
            patch = self.ax3.axvspan(
                self.start_x, self.start_x, color="yellow", alpha = 0.5
            )  # Initialize a translucent rectangle
            self.rectangles.append(patch)
            print(f"Selection started at: {self.start_x:.2f} seconds")
            self.motion_event = self.canvas.mpl_connect("motion_notify_event", self.on_drag)
            self.release_event = self.canvas.mpl_connect("button_release_event", self.on_release)

    def on_drag(self, event):
        # Update the selection rectangle as the mouse moves
        if (event and event.xdata is not None):
            if ((event.inaxes == self.ax3) and event.xdata < 5):
                self.end_x = max(0, event.xdata)  # Update the current x-coordinate
                patch = self.rectangles[-1]
                x0 = min(self.start_x, self.end_x)
                width = abs(self.end_x - self.start_x)

                # Update the rectangle's position and width
                patch.set_x(x0)
                patch.set_width(width)
                self.canvas.draw()
            elif ((event.xdata > 5) or (event.xdata < 0)):
                self.end_x = 5
                self.canvas.mpl_disconnect(self.press_event)
                self.canvas.mpl_disconnect(self.motion_event)
                self.on_release(event)
            

    def on_release(self, event):
        # Finalize the selection when the mouse button is released
        if event and event.xdata is not None:
            self.end_x = max(0, event.xdata)  # Record the ending x-coordinate
            self.end_x = min(self.end_x, 4.9989)
        print(f"Selection ended at: {self.end_x:.2f} seconds")

        print(f"Selected region: {min(self.start_x, self.end_x):.2f} to {max(self.start_x, self.end_x):.2f} seconds")
        self.itr_log.append([int(min(self.start_x*self.sr, self.end_x*self.sr)), int(max(self.start_x*self.sr, self.end_x*self.sr))])
        self.canvas.mpl_disconnect(self.press_event)
        self.canvas.mpl_disconnect(self.motion_event)
        self.canvas.mpl_disconnect(self.release_event)

        self.canvas.draw()

    def update_playback_bar(self):
        if pygame.mixer.music.get_busy() and not self.isPaused:
            # Get the current playback position in milliseconds and convert to seconds
            self.pos = self.new_time + pygame.mixer.music.get_pos() / 1000.0
            if (self.pos < 5):
                self.playback_bar.remove()  # Remove the old bar
                # Update the playback bar
                self.playback_bar = self.ax3.axvspan(self.pos, self.pos + 0.001, color="gray")
                self.canvas.draw()

                self.slider.config(value = self.pos)
                # Schedule the next update
                # Check if music is still playing
                self.after(70, self.update_playback_bar)  # Update every 50ms
        elif not self.isPaused:
            self.pos = 0
            self.slider.config(value = self.pos)
            self.playback_bar.remove()  # Remove the old bar
            # Update the playback bar
            self.playback_bar = self.ax3.axvspan(self.pos, self.pos + 0.001, color="gray")
            self.canvas.draw()
        if not pygame.mixer.music.get_busy() and self.slider.get() > 4.8:
            self.new_time = 0

    def refine(self):
        self.counter += 1

        # Redraw the canvas to update the plot
        self.complete_log.append(self.dir)
        self.complete_log.append(self.itr_log)
        print("Iteration log: ", self.itr_log)
        print("Complete log: ", self.complete_log)
        self.itr_log = []

        if (((self.var.get() == "1-25") and (self.counter == 26))
            or ((self.var.get() == "26-50") and (self.counter == 51))
            or ((self.var.get() == "51-75") and (self.counter == 76))
            or ((self.var.get() == "76-100") and (self.counter == 101))
            or ((self.var.get() == "101-125") and (self.counter == 126))
            or ((self.var.get() == "126-150") and (self.counter == 151))
            or ((self.var.get() == "151-175") and (self.counter == 176))
            or ((self.var.get() == "176-200") and (self.counter == 201))
            or ((self.var.get() == "201-225") and (self.counter == 226))
            or ((self.var.get() == "226-250") and (self.counter == 251))
            ):
            self.done()
        else:
            self.setup_directory()
            self.prepare_audio()

            self.ax.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.draw_waveform()
            self.canvas.draw()
            if (self.var.get() == "1-25"):
                self.itr_label.config(text=f"Example {self.counter}/25")

            elif (self.var.get() == "26-50"):
                self.itr_label.config(text=f"Example {self.counter}/50")

            elif (self.var.get() == "51-75"):
                self.itr_label.config(text=f"Example {self.counter}/75")

            elif ((self.var.get() == "76-100")):
                self.itr_label.config(text=f"Example {self.counter}/100")
            
            elif ((self.var.get() == "101-125")):
                self.itr_label.config(text=f"Example {self.counter}/125")
            
            elif ((self.var.get() == "126-150")):
                self.itr_label.config(text=f"Example {self.counter}/150")
            
            elif ((self.var.get() == "151-175")):
                self.itr_label.config(text=f"Example {self.counter}/175")
            
            elif ((self.var.get() == "176-200")):
                self.itr_label.config(text=f"Example {self.counter}/200")
            
            elif ((self.var.get() == "201-225")):
                self.itr_label.config(text=f"Example {self.counter}/225")
            
            elif ((self.var.get() == "226-250")):
                self.itr_label.config(text=f"Example {self.counter}/250")
            
            
            self.new_time = 0
            self.slider.config(value=0)

    def done(self):
        # self.complete_log.append(self.itr_log)
        print("Iteration log: ", self.itr_log)
        print("Complete log: ", self.complete_log)
        self.itr_log = []
        if (self.var.get() == "1-25"):
            file_path = 'log1_25.json'

        elif (self.var.get() == "26-50"):
            file_path = 'log26-50.json'
            
        elif (self.var.get() == "51-75"):
            file_path = 'log51-75.json'

        elif (self.var.get() == "76-100"):
            file_path = 'log76-100.json'

        elif (self.var.get() == "101-125"):
            file_path = 'log101-125.json'

        elif (self.var.get() == "126-150"):
            file_path = 'log126-150.json'

        elif (self.var.get() == "151-175"):
            file_path = 'log151-175.json'

        elif (self.var.get() == "176-200"):
            file_path = 'log176-200.json'

        elif (self.var.get() == "201-225"):
            file_path = 'log201-225.json'

        elif (self.var.get() == "226-250"):
            file_path = 'log226-250.json'

        # Write data to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(self.complete_log, json_file, indent=4)

        print(f"Data has been written to {file_path}")

        if hasattr(self, 'slider') and self.slider.winfo_exists():
            self.slider.destroy()

        self.quit()  # This properly ends the mainloop and closes the app
        self.destroy()


frame = MainFrame()
frame.mainloop()