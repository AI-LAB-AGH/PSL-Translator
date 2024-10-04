import os
import csv
import json
import cv2
import torch
from rtmpose import RTMPoseDetector
import random
import numpy as np


class LandmarkExtractor:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def __call__(self, img):
        return self.transform(self.model, img)

def augment_sample(sample):
    """Augment sample by rotating and translating frames randomly."""
    # na razie odpuszczam
    # Rotate frames randomly
    if random.random() < 0.6: 
        angle = random.randint(-5, 5)
        sample = [cv2.rotate(frame, angle) for frame in sample]

    # Shift frames randomly
    if random.random() < 0.9:
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sample = [cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0])) for frame in sample]

    return sample

def augment_with_frame_modification(sample, remove_probability=0.2, duplicate_probability=0.2):
    """Augment sample by randomly removing or duplicating frames."""
    augmented_sample = []
    for frame in sample:
        if random.random() > remove_probability:  # Keep the frame with some probability
            augmented_sample.append(frame)
            if random.random() < duplicate_probability:  # Optionally duplicate the frame
                augmented_sample.append(frame)
    
    # Ensure at least one frame remains in the sample
    if len(augmented_sample) == 0:
        augmented_sample.append(random.choice(sample))
    
    return augmented_sample

def augment_landmarks(landmarks, shift_range: float = 0.01):
    """Augment landmarks by shifting each one randomly within a specified range."""
    for i in range(len(landmarks)):
        for j in range(landmarks[i].shape[0]):  # Iterate over all landmarks in the frame
            # Generate random shift for each landmark
            dx = random.uniform(-shift_range, shift_range)
            dy = random.uniform(-shift_range, shift_range)

            # Apply the shift
            landmarks[i][j, 0] += dx
            landmarks[i][j, 1] += dy

            # Clip to [0, 1] to ensure the coordinates remain valid
            landmarks[i][j, 0] = np.clip(landmarks[i][j, 0], 0, 1)
            landmarks[i][j, 1] = np.clip(landmarks[i][j, 1], 0, 1)

    return landmarks


def extract_landmarks_with_RTMP(model, sample):
    landmarks = []
    hand_left_indices = range(91, 112) 
    hand_right_indices = range(112, 133) 
    i = 0
    for frame in sample:
        i += 1
        h, w = frame.shape[:2]
        result = model(frame)
        result[:, 0] /= w
        result[:, 1] /= h
        
        hands_visible = (
            any(result[idx, 0] != 0 or result[idx, 1] != 0 for idx in hand_left_indices) or
            any(result[idx, 0] != 0 or result[idx, 1] != 0 for idx in hand_right_indices)
        )
        
        for idx in range(result.shape[0]):
            x, y = int(result[idx, 0] * w), int(result[idx, 1] * h)
            if idx in hand_left_indices:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue for left hand
            elif idx in hand_right_indices:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red for right hand
        
        # if hands_visible:
        #     landmarks.append(result)
        # else:
        #     cv2.putText(frame, "Hands not visible", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        landmarks.append(result)
        
        cv2.imshow('Landmarks', frame)  # no-spell-check
        cv2.waitKey(1)
    
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


def preprocess_directory(root_dir: str, tgt_dir: str, annotations: dict, label_map: dict, extractor: LandmarkExtractor, augment: bool):
    data = []
    num_dirs = len(os.listdir(root_dir))
    
    os.makedirs(tgt_dir, exist_ok=True)
    
    for i, dir in enumerate(os.listdir(root_dir)):
        path = os.path.join(root_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [cv2.imread(os.path.join(path, frame)) for frame in frames]

        if label_map is not None:
            annotation_value = annotations.get(dir)
            if annotation_value is None:
                print(f"Warning: No annotation found for directory: {dir}")
                continue

            label = label_map.get(annotation_value)
            if label is None:
                print(f"Warning: No label found for annotation: {annotation_value} in directory: {dir}")
                continue 
        else:
            label = annotations.get(dir)
            if label is None:
                print(f"Warning: No annotation found for directory: {dir}")
                continue 

        landmarks = extractor(sample)
        if len(landmarks) > 0:
            data.append((label, landmarks))
        else:
            print(f"Warning: No landmarks found for directory: {dir}")
        
        if augment:
            # Landmarks modification
            augmented_landmarks = extractor(sample)  
            augmented_landmarks = augment_landmarks(augmented_landmarks) 
            if len(augmented_landmarks) > 0: 
                data.append((label, augmented_landmarks))
            else:
                print(f"Warning: No landmarks found for directory: {dir} after augmentation")
            
            # Frames modification augmentations
            modified_sample = augment_with_frame_modification(sample)
            modified_landmarks = extractor(modified_sample)
            if len(modified_landmarks) > 0:
                data.append((label, modified_landmarks))
            else:
                print(f"Warning: No landmarks found for directory: {dir} after augmentation")
            
            # Frame and landmarks modification
            modified_sample = augment_with_frame_modification(sample)
            modified_landmarks = extractor(modified_sample)
            modified_landmarks = augment_landmarks(modified_landmarks)
            if len(modified_landmarks) > 0:
                data.append((label, modified_landmarks))
            else:
                print(f"Warning: No landmarks found for directory: {dir} after augmentation")
        
        print(f'\rDirectory {i+1}/{num_dirs} processed', end='')

        if i % 100 == 0:
            torch.save(data, os.path.join(tgt_dir, 'data.pth'))

    torch.save(data, os.path.join(tgt_dir, 'data.pth'))
    data.clear()


def prepare_dataset(root_dir: str, tgt_dir: str, extractor) -> None:
    labels = os.path.join(root_dir, 'labels.json')
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    tgt_train_dir = os.path.join(tgt_dir, 'train')
    tgt_test_dir = os.path.join(tgt_dir, 'test')

    (annotations_train, annotations_test) = get_annotations(root_dir)

    preprocess_directory(train_dir, tgt_train_dir, annotations_train, label_map, extractor, augment=True)
    preprocess_directory(test_dir, tgt_test_dir, annotations_test, label_map, extractor, augment=False)


def main():
    ### PREPROCESSING PREVIOUSLY EXTRACTED LANDMARKS
    # root_dir = 'data/landmarks'
    # tgt_dir = 'data/landmarks_P'
    # preprocess_landmarks(root_dir, tgt_dir)

    ### PREPROCESSING .JPG FRAMES
    root_dir = 'data/RGB_more_copy'
    tgt_dir = 'data/RGB_more_copy_RTMP'

    ## Mediapipe
    # model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    # transform = extract_landmarks_with_MP

    ## RTMPose
    model = RTMPoseDetector(filepath='preprocessing/landmark_extraction/end2end.onnx')
    transform = extract_landmarks_with_RTMP
    
    extractor = LandmarkExtractor(model, transform)
    prepare_dataset(root_dir, tgt_dir, extractor)


if __name__ == '__main__':
    main()