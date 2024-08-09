import os
import csv
import cv2
import json
import torch
import numpy as np
import mediapipe as mp
from skimage import io
from rtmpose import RTMPoseDetector

def preprocess_landmarks(root_dir: str, tgt_dir: str) -> None:
    labels = os.path.join(root_dir, 'labels.json')
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    target_train_dir = os.path.join(tgt_dir, 'train')
    target_test_dir = os.path.join(tgt_dir, 'test')

    with open(os.path.join(root_dir, 'annotations_train.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_train = {row[0]: row[1] for row in reader}
    f.close()
    with open(os.path.join(root_dir, 'annotations_test.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_test = {row[0]: row[1] for row in reader}
    f.close()

    train = []
    for dir in os.listdir(train_dir):
        path = os.path.join(train_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [np.load(os.path.join(path, frame)) for frame in frames]

        left, right = [], []
        for frame in sample:
            if np.all([_ == 0. for _ in frame[:63]]):
                left.append(torch.tensor(np.array([])))
            else:
                left.append(torch.tensor(frame[:63]))

            if np.all([_ == 0. for _ in frame[63:]]):
                right.append(torch.tensor(np.array([])))
            else:
                right.append(torch.tensor(frame[63:]))
            
        label = label_map[annotations_train[dir]]
        train.append((label, left, right))
    torch.save(train, os.path.join(target_train_dir, 'data.pth'))
    train.clear()

    test = []
    for dir in os.listdir(test_dir):
        path = os.path.join(test_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [np.load(os.path.join(path, frame)) for frame in frames]

        left, right = [], []
        for frame in sample:
            if np.all([_ == 0. for _ in frame[:63]]):
                left.append(torch.tensor(np.array([])))
            else:
                left.append(torch.tensor(frame[:63]))

            if np.all([_ == 0. for _ in frame[63:]]):
                right.append(torch.tensor(np.array([])))
            else:
                right.append(torch.tensor(frame[63:]))
            
        label = label_map[annotations_test[dir]]
        test.append((label, left, right))
    torch.save(test, os.path.join(target_test_dir, 'data.pth'))
    test.clear()


class LandmarkExtractor:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def __call__(self, img):
        return self.transform(self.model, img)


def extract_landmarks_with_RTMP(model, sample):
    landmarks = []
    for frame in sample:
        h, w, c = frame.shape
        result = model(frame)
        result[:, 0] /= w
        result[:, 1] /= h
        landmarks.append(result)

    return landmarks


def extract_landmarks_with_MP(holistic, sample):
    left = []
    right = []

    for frame in sample:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame)

        keypoints = torch.tensor(np.array([]))
        if results.left_hand_landmarks is not None:
            keypoints = torch.tensor(np.array([[l.x, l.y] for l in results.left_hand_landmarks.landmark]), dtype=torch.float32)
            keypoints = keypoints.view(21 * 2)
        left.append(keypoints)
        
        keypoints = torch.tensor(np.array([]))
        if results.right_hand_landmarks is not None:
            keypoints = torch.tensor(np.array([[l.x, l.y] for l in results.right_hand_landmarks.landmark]), dtype=torch.float32)
            keypoints = keypoints.view(21 * 2)
        right.append(keypoints)

    return (left, right)


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


def preprocess_directory(root_dir: str, tgt_dir: str, annotations: dict, label_map: dict, extractor: LandmarkExtractor):
    data = []
    num_dirs = len(os.listdir(root_dir))
    
    for i, dir in enumerate(os.listdir(root_dir)):
        path = os.path.join(root_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [cv2.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations[dir]]

        landmarks = extractor(sample)
        data.append((label, landmarks))

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

    preprocess_directory(train_dir, tgt_train_dir, annotations_train, label_map, extractor)
    preprocess_directory(test_dir, tgt_test_dir, annotations_test, label_map, extractor)


def inference(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        a = model(img)
        cv2.imshow('Camera', img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            return   
                
    cap.release()
    cv2.destroyAllWindows()


def main():
    ### PREPROCESSING PREVIOUSLY EXTRACTED LANDMARKS
    # root_dir = 'data/landmarks'
    # tgt_dir = 'data/landmarks_P'
    # preprocess_landmarks(root_dir, tgt_dir)

    ### PREPROCESSING .JPG FRAMES
    root_dir = 'data/jester'
    tgt_dir = 'data/jester_RTMP'

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