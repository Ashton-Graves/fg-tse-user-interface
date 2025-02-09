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
import onnxruntime as ort

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

        # Load the model and create InferenceSession for tse_model
        tse_model_path = "tse_model.onnx"
        self.tse_session = ort.InferenceSession(tse_model_path)

        # Load the model and create InferenceSession for refinement model
        refinement_model_path = "refinement_model.onnx"
        self.refinement_session = ort.InferenceSession(refinement_model_path)

        # Allow selection for clips to be used
        self.var = tk.StringVar(value="1-10")  # Default selection

        self.released = False
        self.new_time = 0
        self.counter = 1
        self.isPaused = False  # Initialize isPaused here
        

        self.enr_sr = 44100  # example sample rate for enrollment
        self.mix_sr = 44100  # example sample rate for mixture


        file_nums_array = np.linspace(0, 24, 25, dtype=int)
        strings = ["-10-0", "-10000--10", "0-10", "10-10000"]

        # Use np.tile to repeat the file_nums_array for each string, and np.repeat to repeat the strings accordingly
        dir_array = np.array([f"{s}/{str(num).zfill(4)}" for num in file_nums_array for s in strings])
        print(dir_array)
        np.random.seed(43)

        np.random.shuffle(dir_array)

        # Initialize an empty dictionary to hold your file arrays
        self.files_dict = {}

        # Number of selections (50 per set)
        selection_size = 10

        # Loop to create selections (from 1 to 10)
        for i in range(1, 11):
            start_index = (i - 1) * selection_size
            end_index = i * selection_size
            self.files_dict[f"files{i*10-9}-{i*10}"] = dir_array[start_index:end_index]
            print(start_index, " - ", end_index)

        self.setup_directory()
        self.run_tse_model()
        self.prepare_audio(self.onnx_tse_out_file)

        self.itr_log = []
        self.file_log = []
        self.complete_log = []

        # Label to display the selected value
        self.itr_label = tk.Label(self, text="Example 1/10")
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

        self.addPlayButton = tk.Button(self.buttonframe2, text="▶", font=("Arial", 20), command=self.play)
        self.addPlayButton.grid(row=0, column=0)

        self.addPauseButton = tk.Button(self.buttonframe2, text="⏸", font=("Arial", 20), command=self.pause)
        self.addPauseButton.grid(row=0, column=1)

        self.buttonframe2.pack()

        self.addPlayMix = tk.Button(self, text = "Play mixture.wav", command=self.play_mix)
        self.addPlayMix.pack()

        self.addPlayEnr = tk.Button(self, text = "Play enrollment.wav", command=self.play_enr)
        self.addPlayEnr.pack()

        self.draw_figure()
        
        self.playback_bar = self.ax3.axvspan(0, 0.001, color="gray")

        self.addDoneButton = tk.Button(self, text = "Done", command=self.done)
        self.addDoneButton.pack(side="right")

        self.addRefineButton = tk.Button(self, text = "Refine", command=self.refine)
        self.addRefineButton.pack(side="right")

        self.addExitButton = tk.Button(self, text = "Exit", command=self.exit)
        self.addExitButton.pack(side="right")

        # Create a frame for the buttons at the bottom
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side="bottom", fill="x")
        # Add radio buttons to the frame with grid layout

        self.addButton10 = tk.Radiobutton(bottom_frame, variable=self.var, text="1-10", value="1-10", command=self.mode)
        self.addButton10.grid(row=0, column=0)

        self.addButton20 = tk.Radiobutton(bottom_frame, variable=self.var, text="11-20", value="11-20", command=self.mode)
        self.addButton20.grid(row=0, column=1)
        
        self.addButton30 = tk.Radiobutton(bottom_frame, variable=self.var, text="21-30", value="21-30", command=self.mode)
        self.addButton30.grid(row=0, column=2)
        
        self.addButton40 = tk.Radiobutton(bottom_frame, variable=self.var, text="31-40", value="31-40", command=self.mode)
        self.addButton40.grid(row=0, column=3)
        
        self.addButton50 = tk.Radiobutton(bottom_frame, variable=self.var, text="41-50", value="41-50", command=self.mode)
        self.addButton50.grid(row=0, column=4)
        
        self.addButton60 = tk.Radiobutton(bottom_frame, variable=self.var, text="51-60", value="51-60", command=self.mode)
        self.addButton60.grid(row=1, column=0)
        
        self.addButton70 = tk.Radiobutton(bottom_frame, variable=self.var, text="61-70", value="61-70", command=self.mode)
        self.addButton70.grid(row=1, column=1)
        
        self.addButton80 = tk.Radiobutton(bottom_frame, variable=self.var, text="71-80", value="71-80", command=self.mode)
        self.addButton80.grid(row=1, column=2)
        
        self.addButton90 = tk.Radiobutton(bottom_frame, variable=self.var, text="81-90", value="81-90", command=self.mode)
        self.addButton90.grid(row=1, column=3)
        
        self.addButton100 = tk.Radiobutton(bottom_frame, variable=self.var, text="91-100", value="91-100", command=self.mode)
        self.addButton100.grid(row=1, column=3)

        # Label to display the selected value
        self.label = tk.Label(self, text="You selected: 1-10")
        self.label.pack(pady=10)

########### MODELS ############

    # generates edit mask based on the highlighted regions
    def create_edit_mask(self):
        # generates the next fg_tse_output after an iteration of out model
        # create edit mask
        self.edit_mask = np.zeros(len(self.waveform))
        for seg in self.itr_log:
            print("Highlighted segment:, ", seg)
            start_index = seg[0]
            end_index = seg[1]
            self.edit_mask[start_index:end_index] = 1
            print(f"edit_mask[0:{start_index -1}]: ", self.edit_mask[0:start_index -1])
            
            print(f"edit_mask[{start_index}:{end_index}]: ", self.edit_mask[start_index:end_index])
            print(f"edit_mask[{end_index-1}:]: ", self.edit_mask[end_index-1:])

    def run_refinement_model(self):
        self.num_refinements += 1
        print("self.num_refinements: ", self.num_refinements)

        # If a new file is loaded, the mix input is the tse output. 
        # Otherwise, the mix input is the previous refinement model output
        mix, mix_sr = sf.read(self.mixture)
        mix = mix.reshape(1, 1, -1).astype(np.float32)

        embedding = np.load(self.dir + "/embedding.npy")
        embedding = embedding.reshape(1, -1).astype(np.float32)

        edit_mask = self.edit_mask.reshape(1, 1, -1).astype(np.float32)
        current_state = self.next_state.astype(np.float32)

        print(self.mixture)
        print(self.dir + "/embedding.npy")

        # Run inference
        outputs = self.refinement_session.run(["decoded_output", "next_state"], # outputs
                {"edit_mask": edit_mask, "current_state": current_state, "mixture": mix, "embedding": embedding}) # inputs
        outputs = {self.refinement_session.get_outputs()[i].name: outputs[i] for i in range(len(outputs))}
        print(outputs['decoded_output'].shape)
        print(outputs['next_state'].shape)

        # only use refinement model ouput where edit mask is one, otherwise use previous mix
        self.onnx_refinement_out = outputs['decoded_output'].reshape(-1) * edit_mask.reshape(-1) + (1-edit_mask.reshape(-1))*mix.reshape(-1)
 
        self.onnx_refinement_out_file = "/onnx_refinement_output.wav"
        sf.write(self.dir + self.onnx_refinement_out_file, self.onnx_refinement_out, 16000)

        if (self.num_refinements >= 5):
            print("in")
            self.done()
    
    def run_tse_model(self):
        # indicates that this is a new sample and the refinement model hasn't been run yet for this sample
        self.newSample = True

        # resets the amount of times that the refinement model has been run
        self.num_refinements = 0

        # "Load and preprocess the input image inputTensor"
        mix, mix_sr = sf.read(self.mixture)
        assert mix_sr ==16000
        mix = mix.reshape(1, 1, -1).astype(np.float32)

        embedding = np.load(self.dir + "/embedding.npy")
        embedding = embedding.reshape(1, -1).astype(np.float32)

        print("tse_model current mixture: ", self.mixture)
        print("tse_model current embedding: ", self.dir + "/embedding.npy")
        # Run inference
        outputs = self.tse_session.run(["decoded_output", "next_state"], {"mixture": mix, "embedding": embedding})
        outputs = {self.tse_session.get_outputs()[i].name: outputs[i] for i in range(len(outputs))}
        print(outputs['decoded_output'].shape)
        print(outputs['next_state'].shape)

        self.next_state = outputs['next_state']

        self.onnx_tse_out = outputs['decoded_output'] # to be used for playback, then as input for refinement
        self.onnx_tse_out_file = "/onnx_tse_output.wav" 
        sf.write(self.dir + self.onnx_tse_out_file, outputs['decoded_output'].reshape(-1), 16000)
        
    # Sets up directory based on (1) the radio buttons on bottom left and (2) the user moving on to next sample
    def setup_directory(self):
        self.dir = f"./user_study_data/"
        current_index = (self.counter % 10) - 1
        
        if current_index == -1:  # When counter is a multiple of 50, set index to 49
            current_index = 9

        print("current_index: ", current_index)
        
        if (self.var.get() == "1-10"):
            self.dir += self.files_dict["files1-10"][current_index]
        elif (self.var.get() == "11-20"):
            self.dir += self.files_dict["files11-20"][current_index]
        elif (self.var.get() == "21-30"):
            self.dir += self.files_dict["files21-30"][current_index]
        elif (self.var.get() == "31-40"):
            self.dir += self.files_dict["files31-40"][current_index]
        elif (self.var.get() == "41-50"):
            self.dir += self.files_dict["files41-50"][current_index]
        elif (self.var.get() == "51-60"):
            self.dir += self.files_dict["files51-60"][current_index]
        elif (self.var.get() == "61-70"):
            self.dir += self.files_dict["files61-70"][current_index]
        elif (self.var.get() == "71-80"):
            self.dir += self.files_dict["files71-80"][current_index]
        elif (self.var.get() == "81-90"):
            self.dir += self.files_dict["files81-90"][current_index]
        elif (self.var.get() == "91-100"):
            self.dir += self.files_dict["files91-100"][current_index]

        self.mixture = self.dir + "/mixture.wav"
        self.enrollment = self.dir + "/enrollment.wav"



    # Selects group of file by using radio buttons on bottom left. Resets the counter to the first sample of the group. Updates GUI
    def mode(self):
        # Update the label text with the selected value
        self.label.config(text=f"You selected: {self.var.get()}")
        if (self.var.get() == "1-10"):
            self.counter = 1
        elif (self.var.get() == "11-20"):
            self.counter = 11
        elif (self.var.get() == "21-30"):
            self.counter = 21
        elif (self.var.get() == "31-40"):
            self.counter = 31
        elif (self.var.get() == "41-50"):
            self.counter = 41
        elif (self.var.get() == "51-60"):
            self.counter = 51
        elif (self.var.get() == "61-70"):
            self.counter = 61
        elif (self.var.get() == "71-80"):
            self.counter = 71
        elif (self.var.get() == "81-90"):
            self.counter = 81
        elif (self.var.get() == "91-100"):
            self.counter = 91

        if (self.var.get() == "1-10"):
            self.itr_label.config(text=f"Example {self.counter}/10")

        elif (self.var.get() == "11-20"):
            self.itr_label.config(text=f"Example {self.counter}/20")

        elif (self.var.get() == "21-30"):
            self.itr_label.config(text=f"Example {self.counter}/30")

        elif ((self.var.get() == "31-40")):
            self.itr_label.config(text=f"Example {self.counter}/40")
        
        elif ((self.var.get() == "41-50")):
            self.itr_label.config(text=f"Example {self.counter}/50")
        
        elif ((self.var.get() == "51-60")):
            self.itr_label.config(text=f"Example {self.counter}/60")
        
        elif ((self.var.get() == "61-70")):
            self.itr_label.config(text=f"Example {self.counter}/70")
        
        elif ((self.var.get() == "71-80")):
            self.itr_label.config(text=f"Example {self.counter}/80")
        
        elif ((self.var.get() == "81-90")):
            self.itr_label.config(text=f"Example {self.counter}/90")
        
        elif ((self.var.get() == "91-100")):
            self.itr_label.config(text=f"Example {self.counter}/100")
        

        self.setup_directory()
        self.run_tse_model()
        self.prepare_audio(self.onnx_tse_out_file)


        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.draw_waveform()
        self.canvas.draw()

        self.complete_log = []
        self.file_log = []
        self.itr_log = []

    # Loads the wav designated by the string into the output (bottom graph on app, played by using play/pause buttons)
    def prepare_audio(self, string):
        self.file = self.dir + string
        audio = AudioSegment.from_wav(self.file)
        output_file = str(self.file) + ".ogg"
        audio.export(output_file, format="ogg", codec="libvorbis")
        pygame.mixer.music.load(output_file)
        print("Currently loaded audio: ", output_file)


########### GRAPHICAL INTERFACE ###########

    def draw_waveform(self):
        self.waveform, self.sr = sf.read(self.file)
        time = np.linspace(0, len(self.waveform) / self.sr, num=len(self.waveform))
        # Choose a downsampling factor
        downsample_factor = 50

        self.mix_waveform, self.mix_sr = sf.read(self.mixture)
        mix_time = np.linspace(0, len(self.mix_waveform) / self.mix_sr, num=len(self.mix_waveform))

        mix_downsampled_time = mix_time[::downsample_factor]
        mix_downsampled_waveform = self.mix_waveform[::downsample_factor]
        mix_downsampled_waveform = mix_downsampled_waveform / np.max(np.abs(mix_downsampled_waveform))
        self.ax.plot(mix_downsampled_time, mix_downsampled_waveform, color='red')
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title("mixture.wav")

        # Downsample the time and waveform
        downsampled_time = time[::downsample_factor]
        downsampled_waveform = self.waveform[::downsample_factor]
        downsampled_waveform = downsampled_waveform / np.max(np.abs(downsampled_waveform))
        self.ax3.plot(downsampled_time, downsampled_waveform, color='red')
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax3.set_title("tse_output.wav")

        self.enr_waveform, self.enr_sr = sf.read(self.enrollment)
        enr_time = np.linspace(0, len(self.enr_waveform) / self.enr_sr, num=len(self.enr_waveform))

        enr_downsampled_time = enr_time[::downsample_factor]
        enr_downsampled_waveform = self.enr_waveform[::downsample_factor]
        enr_downsampled_waveform = enr_downsampled_waveform / np.max(np.abs(enr_downsampled_waveform))
        self.ax2.plot(enr_downsampled_time, enr_downsampled_waveform, color='red')
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_title("enrollment.wav")

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


##########  REFINE, MOVE TO NEXT SAMPLE (DONE), AND EXIT APPLICATION #############

    def refine(self):
        self.create_edit_mask()
        self.run_refinement_model()
        self.prepare_audio(self.onnx_refinement_out_file)
        self.ax3.clear()
        self.draw_waveform()
        self.canvas.draw()  
        
        print("Itr log: ", self.itr_log)
        print("File log: ", self.file_log)
        self.file_log.append(self.itr_log)
        self.itr_log = []

    def done(self):
        print("done")
        self.counter += 1

        # Redraw the canvas to update the plot
        self.complete_log.append(self.dir)
        self.complete_log.append(self.file_log)
        print("File log: ", self.file_log)
        print("Complete log: ", self.complete_log)
        self.file_log = []

        if (((self.var.get() == "1-10") and (self.counter == 11))
            or ((self.var.get() == "11-20") and (self.counter == 21))
            or ((self.var.get() == "21-30") and (self.counter == 31))
            or ((self.var.get() == "31-40") and (self.counter == 41))
            or ((self.var.get() == "41-50") and (self.counter == 51))
            or ((self.var.get() == "51-60") and (self.counter == 61))
            or ((self.var.get() == "61-70") and (self.counter == 71))
            or ((self.var.get() == "71-80") and (self.counter == 81))
            or ((self.var.get() == "81-90") and (self.counter == 91))
            or ((self.var.get() == "91-100") and (self.counter == 101))
            ):
            self.exit()
        else:
            print("done")
            self.setup_directory()
            self.run_tse_model()
            self.prepare_audio(self.onnx_tse_out_file)

            self.ax.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.draw_waveform()
            self.canvas.draw()
            if (self.var.get() == "1-10"):
                self.itr_label.config(text=f"Example {self.counter}/10")

            elif (self.var.get() == "11-20"):
                self.itr_label.config(text=f"Example {self.counter}/20")

            elif (self.var.get() == "21-30"):
                self.itr_label.config(text=f"Example {self.counter}/30")

            elif ((self.var.get() == "31-40")):
                self.itr_label.config(text=f"Example {self.counter}/40")
            
            elif ((self.var.get() == "41-50")):
                self.itr_label.config(text=f"Example {self.counter}/50")
            
            elif ((self.var.get() == "51-60")):
                self.itr_label.config(text=f"Example {self.counter}/60")
            
            elif ((self.var.get() == "61-70")):
                self.itr_label.config(text=f"Example {self.counter}/70")
            
            elif ((self.var.get() == "71-80")):
                self.itr_label.config(text=f"Example {self.counter}/80")
            
            elif ((self.var.get() == "81-90")):
                self.itr_label.config(text=f"Example {self.counter}/90")
            
            elif ((self.var.get() == "91-100")):
                self.itr_label.config(text=f"Example {self.counter}/100")
            
            
            self.new_time = 0
            self.slider.config(value=0)

    def exit(self):
        self.complete_log.append(self.file_log)
        print("Iteration log: ", self.file_log)
        print("Complete log: ", self.complete_log)
        self.file_log = []
        if (self.var.get() == "1-10"):
            file_path = 'log1_10.json'

        elif (self.var.get() == "11-20"):
            file_path = 'log11_20.json'
            
        elif (self.var.get() == "21-30"):
            file_path = 'log21_30.json'

        elif (self.var.get() == "31-40"):
            file_path = 'log31_40.json'

        elif (self.var.get() == "41-50"):
            file_path = 'log41_50.json'

        elif (self.var.get() == "51-60"):
            file_path = 'log51_60.json'

        elif (self.var.get() == "61-70"):
            file_path = 'log61_70.json'

        elif (self.var.get() == "71-80"):
            file_path = 'log71_80.json'

        elif (self.var.get() == "81-90"):
            file_path = 'log81_90.json'

        elif (self.var.get() == "91-100"):
            file_path = 'log91_100.json'

        # Write data to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(self.complete_log, json_file, indent=4)

        print(f"Data has been written to {file_path}")

        if hasattr(self, 'slider') and self.slider.winfo_exists():
            self.slider.destroy()

        self.quit()  # This properly ends the mainloop and closes the app
        self.destroy()

############# PLAY AUDIO ############

    def slide(self, x):
        self.new_time = self.slider.get()
        pygame.mixer.music.play(start=float(self.new_time))
        pygame.mixer.music.pause()
        self.playback_bar.remove()  # Remove the old bar
        # Update the playback bar
        self.playback_bar = self.ax3.axvspan(self.new_time, self.new_time + 0.001, color="gray")
        self.canvas.draw()

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

        
############# MECHANICS FOR SELECT REGIONS ############
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
            if ((self.start_x <= 5) and (self.start_x >= 0)):
                print(self.start_x)
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
        # Finalize the fection when the mouse button is released
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


frame = MainFrame()
frame.mainloop()