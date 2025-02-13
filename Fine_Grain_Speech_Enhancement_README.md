Target Speech Extraction User Interface README
This README was generated by ChatGPT and edited by Ashton Graves

Welcome to the Target Speech Extraction (TSE) User Interface! This tool allows you to annotate audio data by listening to the mixture, enrollment, and target speech output, and highlight relevant regions of the waveform for annotation.

Follow the instructions below to set up and use this interface.

1. Setting Up the Environment

Navigate to the directory containing the setup.sh file, the requirements.txt file, and the 
user_selection.py file.

** Note: It’s recommended to activate your preferred conda environment before running the setup.sh file, as Python dependencies will be installed into the currently active environment. **

Open the terminal and run the following command to allow the setup.sh file to execute:

bash ./setup.sh

or 

chmod +x setup.sh

Now, run the setup file to configure your environment:
./setup.sh

Please reach out to Ashton via email if more issues present themselves with downloading ffmpeg: 
graveash@uw.edu

2. Running the Interface

Once the setup is complete, you can run the interface by executing the following command in the terminal:

python user_selection.py
This will open the Target Speech Extraction User Interface.

3. Using the Interface

Controls:

Use the radio buttons (1-50, 51-100, etc.) at the bottom of the window to switch to a grouping of files that has not been annotated yet. 

*** IMPORTANT. Refer to this Google Sheet to see which files have already been annotated: https://docs.google.com/spreadsheets/d/16i_OWT3Gpn7Ghmd-sF1pVHA2Gqu27HLDCN7HC8u_YOs/edit?usp=sharing. Please update the Google Sheet when you have finished annotating ***


Play/Resume: Click the "▶" button to play or resume the audio.
Pause: Click the "⏸" button to pause the audio.
Play Mixture: Click the "Play mixture.wav" button to play the mixture audio.
Play Enrollment: Click the "Play enrollment.wav" button to play the enrollment audio.
Highlighting Regions:
Highlight: Click the "Highlight" button and then click and drag over the waveform to highlight relevant regions of the audio. You can add multiple highlights.
Undo Highlight: Click the "↺" button to undo the last highlight.
Navigating Between Audio Clips:
Use the "Next" button to go to the next file in your current selection.

Playback Slider:
Use the slider to seek through the waveform. You can drag the slider to move to different parts of the audio and update the playback accordingly.

Avoid Overwriting JSON Files:
Be careful not to overwrite any existing JSON files by annotating the same batch of files that another team member already annotated. Consult the shared Google Sheet to view which files have already been annotated.
