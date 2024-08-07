import numpy as np
import cv2 as cv
import torch
import os
import json
import csv
from skimage import io

def extract_optical_flow(frames: list[np.ndarray]) -> list[np.ndarray]:
    flow = []
    for i in range(1, len(frames)):
        img1 = frames[i-1]
        img2 = frames[i]
        img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
        flow.append(torch.tensor(cv.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 5, 3, 5, 1.2, 0), dtype=torch.int8))
    return flow


def rgb_to_optical_flow_dataset(root_dir, target_dir):
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

    train = []
    total = len(os.listdir(train_dir))
    for i, dir in enumerate(os.listdir(train_dir)):
        path = os.path.join(train_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [io.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations_train[dir]]
        flow = extract_optical_flow(sample)
        train.append((label, flow))
        print(f'\rDirectory {i+1}/{total} processed', end='')
    torch.save(train, os.path.join(target_train_dir, 'data.pth'))
    train.clear()

    # test = torch.load(os.path.join(tgt_test_dir, 'data.pth'))
    test = []
    total = len(os.listdir(test_dir))
    for i, dir in enumerate(os.listdir(test_dir)):
        path = os.path.join(test_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [io.imread(os.path.join(path, frame)) for frame in frames]
        label = label_map[annotations_test[dir]]
        flow = extract_optical_flow(sample)
        test.append((label, flow))
        print(f'\rDirectory {i+1}/{total} processed', end='')
    torch.save(test, os.path.join(target_test_dir, 'data.pth'))
    test.clear()




def main():
    root_dir = os.path.join('data', 'RGB_debug')
    tgt_dir = os.path.join('data', 'RGB_OF')
    rgb_to_optical_flow_dataset(root_dir, tgt_dir)

if __name__ == '__main__':
    main()
