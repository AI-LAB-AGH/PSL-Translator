import numpy as np
import cv2 as cv
import torch
import os
import json
import csv
import argparse
from skimage import io
from typing import Any

def extract_optical_flow(frames: list[np.ndarray]) -> list[np.ndarray]:
    flow = []
    for i in range(1, len(frames)):
        img1 = frames[i-1]
        img2 = frames[i]
        img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
        flow.append(torch.tensor(cv.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 5, 3, 5, 1.2, 0), dtype=torch.float32))
    return flow


def process_data(root_dir: str, target_dir: str, batch_size: int, annotations: dict, label_map: Any):
    data = []
    total = len(os.listdir(root_dir))
    for i, dir in enumerate(os.listdir(root_dir)):
        if i % batch_size == 0:
            first_in_batch = i
        path = os.path.join(root_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [io.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations[dir]]
        flow = extract_optical_flow(sample)
        data.append((label, flow))
        print(f'\rDirectory {i+1}/{total} processed', end='')
        if (i + 1) % batch_size == 0 or i + 1 == total:
            torch.save(data, os.path.join(target_dir, f'data_{first_in_batch}_{i}.pth'))
            data.clear()


def rgb_to_optical_flow_dataset(root_dir: str, target_dir: str, batch_size: int):
    labels = os.path.join(root_dir, 'labels.json')
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    target_train_dir = os.path.join(target_dir, 'train')
    target_test_dir = os.path.join(target_dir, 'test')

    with open(os.path.join(root_dir, 'annotations_train.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_train = {row[0]: row[1] for row in reader}
    f.close()
    with open(os.path.join(root_dir, 'annotations_test.csv'), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        annotations_test = {row[0]: row[1] for row in reader}
    f.close()

    process_data(train_dir, target_train_dir, batch_size, annotations_train, label_map)
    process_data(test_dir, target_test_dir, batch_size, annotations_test, label_map)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10, help='Number of video samples to be stored in each .pth file')
    parser.add_argument('--root_dir', type=str, default=os.path.join('data', 'RGB_debug'), help='Root directory containing the raw video dataset (.jpg)')
    parser.add_argument('--target_dir', type=str, default=os.path.join('data', 'RGB_OF'), help='Directory in which to save optical flow sequences (.pth)')

    return parser.parse_args()


def main():
    # get CLI arguments
    args = get_args()
    batch_size = args.batch_size
    root_dir = args.root_dir
    tgt_dir = args.target_dir

    rgb_to_optical_flow_dataset(root_dir, tgt_dir, batch_size=batch_size)

if __name__ == '__main__':
    main()
