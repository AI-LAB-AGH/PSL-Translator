import os
import csv
import cv2
from rtmpose import RTMPoseDetector
import mediapipe as mp
import matplotlib.pyplot as plt

MP_POSE = mp.solutions.pose.Pose(static_image_mode=True)
MP_HANDS = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
RTMP = RTMPoseDetector(filepath='preprocessing/end2end.onnx')


def visualize_landmarks(image, keypoints):
    """
    Visualize landmarks on the image using Matplotlib.

    :param image_path: Path to the image file.
    :param keypoints: List of keypoints to overlay on the image.
    """
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for x, y, _ in keypoints:
        if x != 0 and y != 0:
            plt.scatter(x * w, y * h, c='red', s=10)

    plt.axis('off')
    plt.show()

def extract_landmarks_with_MP(sample):
    landmarks = []

    for frame in sample:
        keypoints = []
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = MP_POSE.process(image_rgb)
        hands_results = MP_HANDS.process(image_rgb)

        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([[0, 0, 0]] * 33)

        left_hand = [[0, 0, 0]] * 21
        right_hand = [[0, 0, 0]] * 21
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                hand_type = handedness.classification[0].label
                if hand_type == "Left":
                    left_hand = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                elif hand_type == "Right":
                    right_hand = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]

        keypoints.extend(left_hand)
        keypoints.extend(right_hand)
        landmarks.append(keypoints)        

    # Returns 75x3 list (33 pose, 21 left, 21 right) (xyz) []
    return landmarks

def extract_landmarks_with_RTMP(sample):
    landmarks = []
    for frame in sample:
        h, w, c = frame.shape
        result = RTMP.process(frame)
        result[:, 0] /= w
        result[:, 1] /= h
        landmarks.append(result)

    # Returns 133x2 list (18 body, 6 feet, 67 face, 21 left, 21 right) (xy) [https://user-images.githubusercontent.com/100993824/227770977-c8f00355-c43a-467e-8444-d307789cf4b2.png]
    return landmarks

def get_annotations(root_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(root_dir, 'annotations_train.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_train = {row[0]: row[1] for row in reader}
    f.close()

    with open(os.path.join(root_dir, 'annotations_test.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_test = {row[0]: row[1] for row in reader}
    f.close()

    return (annotations_train, annotations_test)