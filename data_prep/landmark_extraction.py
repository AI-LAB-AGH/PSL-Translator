import numpy as np


class ExtractLandmarks:
    def __init__(self, holistic):
        self.holistic = holistic

    def __call__(self, sample: list) -> list:
        # Landmarks already extracted
        if len(sample[0].shape) != 3:
            return sample

        processed = []
        for frame in sample:
            results = self.holistic.process(frame)
            keypoints = np.array([])
            for landmark_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if landmark_list is not None:
                    for landmark in landmark_list.landmark:
                        keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
                else:
                    keypoints = np.append(keypoints, np.zeros(21 * 3))
            processed.append(keypoints.copy())
        return processed
