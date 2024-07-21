import os
import csv
import json
import torch
import numpy as np
import mediapipe as mp
from skimage import io

def PreprocessLandmarks(root_dir: str, tgt_dir: str) -> None:
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


def PreprocessFrames(root_dir: str, tgt_dir: str) -> None:
    labels = os.path.join(root_dir, 'labels.json')
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    tgt_train_dir = os.path.join(tgt_dir, 'train')
    tgt_test_dir = os.path.join(tgt_dir, 'test')
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

    with open(os.path.join(root_dir, 'annotations_train.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_train = {row[0]: row[1] for row in reader}
    f.close()
    with open(os.path.join(root_dir, 'annotations_test.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_test = {row[0]: row[1] for row in reader}
    f.close()

    # train = torch.load(os.path.join(tgt_train_dir, 'data.pth'))
    train = []
    total = len(os.listdir(train_dir))
    for i, dir in enumerate(os.listdir(train_dir)):
        path = os.path.join(train_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [io.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations_train[dir]]
        (left, right) = ExtractLandmarks(holistic, sample)
        train.append((label, left, right))
        print(f'\rDirectory {i+1}/{total} processed', end='')
    torch.save(train, os.path.join(tgt_train_dir, 'data.pth'))
    train.clear()

    # test = torch.load(os.path.join(tgt_test_dir, 'data.pth'))
    test = []
    total = len(os.listdir(test_dir))
    for i, dir in enumerate(os.listdir(test_dir)):
        path = os.path.join(test_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [io.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations_test[dir]]
        (left, right) = ExtractLandmarks(holistic, sample)
        test.append((label, left, right))
        print(f'\rDirectory {i+1}/{total} processed', end='')
    torch.save(test, os.path.join(tgt_test_dir, 'data.pth'))
    test.clear()
    

def ExtractLandmarks(holistic, sample):
    left = []
    right = []

    for frame in sample:
        results = holistic.process(frame)

        keypoints = torch.tensor(np.array([]))
        if results.left_hand_landmarks is not None:
            keypoints = torch.tensor(np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]), dtype=torch.float32)
            keypoints = keypoints.view(21 * 3)
        left.append(keypoints)
        
        keypoints = torch.tensor(np.array([]))
        if results.right_hand_landmarks is not None:
            keypoints = torch.tensor(np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]), dtype=torch.float32)
            keypoints = keypoints.view(21 * 3)
        right.append(keypoints)

    return (left, right)


def main():
    # PREPROCESSING .JPG FRAMES
    root_dir = 'data/jester'
    tgt_dir = 'data/jester_P'
    PreprocessFrames(root_dir, tgt_dir)

    # PREPROCESSING PREVIOUSLY EXTRACTED LANDMARKS
    # root_dir = 'data/landmarks'
    # tgt_dir = 'data/landmarks_P'
    # PreprocessLandmarks(root_dir, tgt_dir)

if __name__ == '__main__':
    main()