import os
import cv2
import mediapipe as mp
import keyboard


def count_samples() -> int:
    train_dirs = os.listdir(os.path.join("../data/landmarks/train/"))
    test_dirs = os.listdir(os.path.join("../data/landmarks/test"))
    total_dirs = len(train_dirs) + len(test_dirs)
    return total_dirs
