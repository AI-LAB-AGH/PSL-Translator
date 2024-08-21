import numpy as np
from pynput import keyboard

from helpers import *
from actions import ACTION_TO_IDX


"""
Run this file on its own to collect new data samples.
For this to work you need to create the following directory structure (data is in .gitignore):

- PSL-Translator/
  - data/
    - RGB/
      - train/
      - test/
      - annotations_train.csv
      - annotations_test.csv

annotations_train.csv should have 2 columns 'sample_idx' and 'class_idx' respectively
"""

space_pressed = False
right_arrow_pressed = False
left_arrow_pressed = False

def on_press(key):
    global space_pressed, right_arrow_pressed, left_arrow_pressed

    if key == keyboard.Key.space:
        space_pressed = True
    elif key == keyboard.Key.right:
        right_arrow_pressed = True
    elif key == keyboard.Key.left:
        left_arrow_pressed = True

def on_release(key):
    global space_pressed, right_arrow_pressed, left_arrow_pressed
    if key == keyboard.Key.space:
        space_pressed = False
    if key == keyboard.Key.right:
        right_arrow_pressed = False
    elif key == keyboard.Key.left:
        left_arrow_pressed = False


def annotate_sample(sample_num: int, action: str, subset: str):
    """
    subset is supposed to only take values {'train', 'test'}
    """
    with open(os.path.join(PATH, f'annotations_{subset}.csv'), 'a') as f:
        f.write(f'{sample_num},{action}\n')


def save_image(cap, subset, sample_num, frame, path):
    success, image = cap.read()
    if not success:
        print("Failed to capture image.")
        return False

    if frame is not None:
        cv2.imwrite(os.path.join(path, subset, str(sample_num), f'{frame}.jpg'), image)


def main():
    actions = np.array(list(ACTION_TO_IDX.keys()))
    sequences = 100  # max number of samples to record
    frames = 100  # max number of frames per sample

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return

    n_samples = count_samples()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        action_idx = 0
        action = actions[action_idx]

        # Start the keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        for sequence in range(sequences):
            while not space_pressed:
                image = draw_landmarks(cap, holistic)
                add_window_text(image, action)
                cv2.imshow('Camera', image)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                if right_arrow_pressed:
                    if action_idx < len(actions) - 1:
                        action_idx += 1
                    else:
                        action_idx = 0
                    action = actions[action_idx]
                    cv2.waitKey(100)

                if left_arrow_pressed:
                    if action_idx > 0:
                        action_idx -= 1
                    else:
                        action_idx = len(actions) - 1
                    action = actions[action_idx]
                    cv2.waitKey(100)

            if n_samples % 5 == 0:
                create_directory(f'test/{n_samples}')
                subset = 'test'
            else:
                create_directory(f'train/{n_samples}')
                subset = 'train'

            print(f"Collection of data for: {action} sequence no: {sequence}.")

            # Give the user 3 seconds to get their hand of the space bar and prepare to show the gesture
            # as soon as the recording commences
            countdown(cap, holistic)

            for frame in range(frames):
                save_image(cap, subset, n_samples, frame, PATH)
                image = draw_landmarks(cap, holistic)
                cv2.putText(image, f"Collecting frames. Action: {action} frame no: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f"Press ESC to stop recording", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)

                # cv2.waitKey essentially serves as a sleep function here, it causes a slight delay
                if cv2.waitKey(1) == 113:  # q to stop recording
                    break

            annotate_sample(n_samples, action, subset)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

            n_samples += 1

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()


if __name__ == "__main__":
    main()
