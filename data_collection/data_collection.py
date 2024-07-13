import mediapipe as mp
import numpy as np
import keyboard

from helpers import *
from actions import ACTION_TO_IDX


def annotate_sample(sample_num: int, class_idx: int):
    with open(os.path.join(PATH, 'annotations.csv'), 'a') as f:
        f.write(f'{sample_num}, {class_idx}\n')


def draw_landmarks_and_save_image(cap, holistic, subset, sample_num, frame, path):
    success, image = cap.read()
    if not success:
        print("Failed to capture image.")
        return False

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if frame is not None:
        cv2.imwrite(os.path.join(path, subset, str(sample_num), f'{frame}.jpg'), image)

    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)

    return image


def main():
    actions = np.array(list(ACTION_TO_IDX.keys()))
    sequences = 100  # max number of samples to record
    frames = 30 # frames per sample

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return

    n_samples = count_samples()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        action_idx = 0
        action = actions[action_idx]
        for sequence in range(sequences):
            while not keyboard.is_pressed(' '):
                image = draw_landmarks_and_save_image(cap, holistic, action, sequence, None, PATH)
                add_window_text(image, action)
                cv2.imshow('Camera', image)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if keyboard.is_pressed('right'):
                    action_idx += 1
                    action = actions[action_idx]
                if keyboard.is_pressed('left'):
                    action_idx -= 1
                    action = actions[action_idx]
            if n_samples % 5 == 0:
                create_directory(f'test/{n_samples}')
                subset = 'test'
            else:
                create_directory(f'train/{n_samples}')
                subset = 'train'
            print(f"Collection of data for: {action} sequence no: {sequence}.")
            for frame in range(frames):
                image = draw_landmarks_and_save_image(cap, holistic, subset, n_samples, frame, PATH)
                cv2.putText(image, f"Collecting frames. Action: {action} frame no: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
            annotate_sample(n_samples, ACTION_TO_IDX[action])

            if cv2.waitKey(1) & 0xFF == 27:
                break
            n_samples += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
