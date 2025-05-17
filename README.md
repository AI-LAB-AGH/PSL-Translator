# PSL-Translator

This is our attempt at creating a model able to tackle the task of Continuous Sign Language Translation (CSLT) - we aim to translate Polish sign language into Polish in real time, basing only on RGB data from a common webcam.

## Modules

Use everything from the root project directory.

### Data

Leave it alone

### DVC

Meant for managing the dataset. Lets you record new samples, push them to a remote repository and pull existing samples (choice specified in the `command` argument). <br >
**Usage:** `py dvc/main.py --command [rec | pull | push]`

### Preprocessing

Meant for preprocessing the RGB dataset. Extracts landmarks from samples using MediaPipe or RTMPose (choice specified in the `transform` argument)
**Usage:** `py preprocessing/main.py --transform [MP | RTMP]`

### Translation

Leave it alone for now

### Main

Meant for running training and inference.
**Usage:** `py main.py [args]`
