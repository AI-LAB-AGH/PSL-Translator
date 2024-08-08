import torch
import numpy as np
import cv2

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

    def __call__(self, sample: list[np.array], confidence=0.5) -> tuple[list, list]:
        # Landmarks already extracted (used during training)
        if type(sample) == tuple:
            return sample

        left = []
        right = []

        for frame in sample:
            hands, scores = self.model(frame)
            h, w, c = frame.shape

            if np.all(scores[0] < confidence):
                l = torch.tensor(np.array([]))
            else:
                l = torch.tensor(hands[0])
                l[:, 0] /= w
                l[:, 1] /= h
                l = l.view(21 * 2)
            left.append(l)

            if np.all(scores[1] < confidence):
                r = torch.tensor(np.array([]))
            else:
                r = torch.tensor(hands[1])
                r[:, 0] /= w
                r[:, 1] /= h
                r = r.view(21 * 2)
            right.append(r)
            
        return (left, right)


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
    
class NormalizeDistances:
    def __call__(self, sample: tuple[torch.tensor, torch.tensor]) -> tuple[torch.tensor, torch.tensor]:
         left = sample[0]
         for frame in left:
            if frame.shape[0] != 0:
                frame = torch.reshape(frame, (21, 2))
                w = torch.max(frame[:, 0]) - torch.min(frame[:, 0])
                h = torch.max(frame[:, 1]) - torch.min(frame[:, 1])
                frame[:, 0] /= w
                frame[:, 1] /= h
                frame = frame.view(21 * 2)

         right = sample[0]
         for frame in right:
            if frame.shape[0] != 0:
                frame = torch.reshape(frame, (21, 2))
                w = torch.max(frame[:, 0]) - torch.min(frame[:, 0])
                h = torch.max(frame[:, 1]) - torch.min(frame[:, 1])
                frame[:, 0] /= w
                frame[:, 1] /= h
                frame = frame.view(21 * 2)

         return (left, right)


class ExtractOpticalFlow:
    def __call__(self, prev: torch.tensor, curr: torch.tensor) -> torch.tensor:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        flow = torch.tensor(cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 5, 3, 5, 1.2, 0), dtype=torch.float32)
        return flow
