import os
import cv2
import mediapipe as mp
import keyboard


def create_directory(name):
    if not os.path.exists(os.path.join("data_updated/", action)):
        os.makedirs(os.path.join("data_updated/", action))

    sequence_list = os.listdir(os.path.join("data_updated/", action))
    sequences_no = len(sequence_list)

    os.makedirs(os.path.join("data_updated/", action, str(sequences_no)), exist_ok=True)
