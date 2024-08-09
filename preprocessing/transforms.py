import torch
import numpy as np

class ComputeDistNetWithMovement:
    def __call__(self, sample: tuple[list, list]) -> tuple[torch.tensor, torch.tensor]:
        left = sample[0]
        source = None
        for frame in range(len(left)):
            left[frame] = torch.tensor(np.array(left[frame]), dtype=torch.float32)
            if left[frame].shape[0] != 0:
                left[frame] = torch.reshape(left[frame], (21, 2))
                left[frame][1:] -= left[frame][0] # Landmark net
                if source is None:
                    source = left[frame][0].clone()
                left[frame][0] -= source # Displacement relative to previous frame
                left[frame][0] /= source # Scale to be a percentage
                source += left[frame][0]
                left[frame] = left[frame].view(21 * 2)

        right = sample[1]
        source = None
        for frame in range(len(right)):
            right[frame] = torch.tensor(np.array(right[frame]), dtype=torch.float32)
            if right[frame].shape[0] != 0:
                right[frame] = torch.reshape(right[frame], (21, 2))
                right[frame][1:] -= right[frame][0]
                if source is None:
                    source = right[frame][0].clone()
                right[frame][0] -= source
                right[frame][0] /= source
                source += right[frame][0]
                right[frame] = right[frame].view(21 * 2)

        return (left, right)
    
    
class ComputeDistNetNoMovement:
    def __call__(self, sample: tuple[list, list]) -> tuple[torch.tensor, torch.tensor]:
        left = sample[0]
        for frame in range(len(left)):
            left[frame] = torch.tensor(np.array(left[frame]), dtype=torch.float32)
            if left[frame].shape[0] != 0:
                left[frame] = torch.reshape(left[frame], (21, 2))
                source = left[frame][0].clone()
                left[frame] -= source
                left[frame] = left[frame].view(21 * 2)

        right = sample[1]
        for frame in range(len(right)):
            right[frame] = torch.tensor(np.array(right[frame]), dtype=torch.float32)
            if right[frame].shape[0] != 0:
                right[frame] = torch.reshape(right[frame], (21, 2))
                source = right[frame][0].clone()
                right[frame] -= source
                right[frame] = right[frame].view(21 * 2)

        return (left, right)


class ExtractLandmarksWithRTMP:
    def __init__(self, model):
        self.model = model

    def __call__(self, sample: list[np.array], confidence=0.4) -> tuple[list, list]:
        # Landmarks already extracted (used during training)
        if type(sample) == tuple:
            return sample

        landmarks = []
        for frame in sample:
            h, w, c = frame.shape
            result = self.model(frame)
            result[:, 0] /= w
            result[:, 1] /= h
            landmarks.append(result)

        return landmarks


class ExtractLandmarksWithMP:
    def __init__(self, holistic):
        self.holistic = holistic

    def __call__(self, sample: list[torch.tensor]) -> tuple[list, list]:
        # Landmarks already extracted (used during training)
        if type(sample) == tuple:
            return sample

        left = []
        right = []
        for frame in sample:
            results = self.holistic.process(frame)
            keypoints = np.array([])

            if results.left_hand_landmarks is not None:
                for landmark in results.left_hand_landmarks.landmark:
                    keypoints = np.append(keypoints, [landmark.x, landmark.y])
            else:
                keypoints = np.append(keypoints, [])
            left.append(keypoints.copy())
            
            keypoints = np.array([])
            if results.right_hand_landmarks is not None:
                for landmark in results.right_hand_landmarks.landmark:
                    keypoints = np.append(keypoints, [landmark.x, landmark.y])
            else:
                keypoints = np.append(keypoints, [])
            right.append(keypoints.copy())

        return (left, right)
    

class ComputeDistances:
    def __call__(self, sample: list[np.array]) -> list[torch.tensor]:
        for i in range(len(sample)):
            sample[i] = torch.tensor(sample[i], dtype=torch.float32)

            source_body = sample[i][0].clone()
            source_face = sample[i][53].clone()
            source_left = sample[i][100].clone()
            source_right = sample[i][121].clone()

            body = sample[i][:17]
            feet = sample[i][17:23]
            face = sample[i][23:91]
            left = sample[i][91:112]
            right = sample[i][112:]
            w_left = torch.max(left[:, 0]) - torch.min(left[:, 0])
            h_left = torch.max(left[:, 1]) - torch.min(left[:, 1])
            w_right = torch.max(right[:, 0]) - torch.min(right[:, 0])
            h_right = torch.max(right[:, 1]) - torch.min(right[:, 1])

            chin2left = left - source_body
            chin2right = right - source_body
            body -= source_body
            face -= source_face
            left -= source_left
            right -= source_right

            if w_left != 0. and h_left != 0:
                left[:, 0] /= w_left
                left[:, 1] /= h_left

            if w_right != 0. and h_right != 0:
                right[:, 0] /= w_right
                right[:, 1] /= h_right

            sample[i] = torch.concat([body, feet, face, left, right, chin2left, chin2right])
            sample[i] = sample[i].view((133 + 42) * 2)

        return sample