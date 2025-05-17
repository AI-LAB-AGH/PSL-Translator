import os
import json
import cv2
import torch
import argparse
from utils import *

def preprocess_directory(root_dir: str, tgt_dir: str, annotations: dict, label_map: dict | None, transform):
    data_path = os.path.join(tgt_dir, 'data.pth')
    data = []
    
    if os.path.getsize(data_path) > 0:
        data = torch.load(data_path, weights_only=False)

    print(f"Processing {root_dir}. Data length: {len(data)}")
    print(f'{root_dir}: {len(data)} directories processed so far')
    num_dirs = len(os.listdir(root_dir))
    
    for i, dir in enumerate(sorted(os.listdir(root_dir), key=lambda name: int(os.path.splitext(name)[0]))):
        if i < len(data):
            continue

        # get sample
        path = os.path.join(root_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [cv2.imread(os.path.join(path, frame)) for frame in frames]

        # convert text label to numeric label
        if label_map is not None:
            label = label_map[annotations[dir]]
        else:
            label = annotations[dir]

        landmarks = transform(sample)
        print(f"Added entry:\b- Label: {label}\n-Landmarks: {landmarks}")
        data.append((label, landmarks))

        print(f'\rDirectory {i+1}/{num_dirs} processed', end='')

        if i % 100 == 0:
            torch.save(data, data_path)

    torch.save(data, data_path)
    data.clear()

def prepare_dataset(root_dir: str, tgt_dir: str, transform) -> None:
    labels = os.path.join(root_dir, 'labels.json')
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    tgt_train_dir = os.path.join(tgt_dir, 'train')
    tgt_test_dir = os.path.join(tgt_dir, 'test')

    (annotations_train, annotations_test) = get_annotations(root_dir)

    preprocess_directory(train_dir, tgt_train_dir, annotations_train, label_map, transform)
    preprocess_directory(test_dir, tgt_test_dir, annotations_test, label_map, transform)


def main():
    parser = argparse.ArgumentParser(description='Preprocess RGB video frames into landmark data')
    parser.add_argument('--transform', type=str, choices=['MP', 'RTMP'], default='MP',
                      help='Transform type to use: MP (MediaPipe) or RTMP (RTMPose) (default: MP)')
    
    args = parser.parse_args()
    
    root_dir = "data/RGB"
    tgt_dir = f"{root_dir}_{args.transform}"
    transform = extract_landmarks_with_MP if args.transform == 'MP' else extract_landmarks_with_RTMP
    
    prepare_dataset(root_dir, tgt_dir, transform)

if __name__ == '__main__':
    main()