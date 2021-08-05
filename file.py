"""
Reading Sheet Music Plan:
Dataset: Musescore (approx 20-30 images of different slices of each note)
- 10-15 digital screenshots + 10-15 real images


Revised Plan:
1. Use OpenCV blob detection to pick out noteheads and rest symbols
2. Train classifier on dataset (./note_dataset_digital)
    - duration types: sixteenth, eighth, quarater, half, whole + variations
    - pitch range C1-A2
3. Classify images by feeding individual "blobs" to classifier
    - duration
    - pitch
4. Play music based on the classified metrics
    - save to MIDI file?


Idea: 
Problem with idea: "Nuclear weapon on ant hill" -ryan
1. Duration classifier (first layer) will determine the length of the note based off of if whether the hole is filled or not, dot after the note, tail at the end of note (8th/16th)
2. Second layer will determine the letter name of the note based off of staff lines
(Jacob mentioned using a confusor note?)

Petar's tips:
Model 1: Parse sheet music into separate notes in order of left to right, output boxes of cropped areas of the note
- output 2 xy pairs, bottom left and top right corners of box
Model 2: Take in a note and classify it by note pitch and duration

Maintain order in the output of the models so that afterwards we can combine the characteristics of the same note together.
""" 