import os
import cv2


"""
These functions are either purely cosmetic or simple enough to move them here
so as to not to clutter data_collection.py too much.
"""


PATH = os.path.join('data', 'RGB')

def count_samples() -> int:
    train_dirs = os.listdir(os.path.join(PATH, 'train'))
    test_dirs = os.listdir(os.path.join(PATH, 'test'))
    total_dirs = len(train_dirs) + len(test_dirs)
    return total_dirs


def create_directory(name):
    if not os.path.exists(os.path.join(PATH, name)):
        os.makedirs(os.path.join(PATH, name))


def add_window_text(image: cv2.UMat, action: str):
    cv2.putText(image, f"Collecting data for: {action}. Press SPACE to start recording,", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image,
                f"RIGHT ARROW to record next action, LEFT ARROW to record previous action,",
                (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image,
                f"ESC to exit.",
                (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


