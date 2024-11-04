import os
import csv
import json
import cv2
import torch
from rtmpose import RTMPoseDetector


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
    data_path = os.path.join(tgt_dir, 'data.pth')
    data = torch.load(data_path)
    print(f'{root_dir}: {len(data)} directories processed so far')
    num_dirs = len(os.listdir(root_dir))
    
    for i, dir in enumerate(sorted(os.listdir(root_dir), key=lambda a: int(os.path.splitext(a)[0]))):
        if i < len(data):
            continue

        path = os.path.join(root_dir, dir)
        frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
        sample = [cv2.imread(os.path.join(path, frame)) for frame in frames]
        if label_map is not None:
            label = label_map[annotations[dir]]
        else:
            label = annotations[dir]

        landmarks = extractor(sample)
        data.append((label, landmarks))

        print(f'\rDirectory {i+1}/{num_dirs} processed', end='')

        if i % 100 == 0:
            torch.save(data, data_path)

    torch.save(data, data_path)
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


def main():
    ### PREPROCESSING PREVIOUSLY EXTRACTED LANDMARKS
    # root_dir = 'data/landmarks'
    # tgt_dir = 'data/landmarks_P'
    # preprocess_landmarks(root_dir, tgt_dir)

    ### PREPROCESSING .JPG FRAMES
    root_dir = 'data/RGB'
    tgt_dir = 'data/RGB_RTMP'

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