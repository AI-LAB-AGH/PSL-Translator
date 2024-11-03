import torch
import time

class GestureRecognitionHandler:
    def __init__(self, model, label_map, confidence_threshold=0.95, consecutive_frames=6):
        self.model = model
        self.actions = dict([(value, key) for key, value in label_map.items()])
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames = consecutive_frames
        self.confidence_window = []
        self.action_window = []
        self.hands_visible_window = []

        self.hand_left_indices = [idx for idx in range(91, 112)]
        self.hand_right_indices = [idx for idx in range(112, 133)]

        self.model.initialize_cell_and_hidden_state()

        self.last_action = ""
        self.last_action_time = time.time()
        self.action_display_duration = 1

    def process_frame(self, frame, transform):
        x = transform([frame])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)

        output = self.model(x)
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = self.actions[predicted_index.item()]

        result = x[0].clone().detach().numpy()
        hands_visible = (
            any(result[idx, 0] != 0 or result[idx, 1] != 0 for idx in self.hand_left_indices) or
            any(result[idx, 0] != 0 or result[idx, 1] != 0 for idx in self.hand_right_indices)
        )
        self.hands_visible_window.append(hands_visible)

        if len(self.hands_visible_window) > 5:
            self.hands_visible_window.pop(0)

        self.confidence_window.append(confidence.item())
        self.action_window.append(predicted_action)

        if len(self.confidence_window) > self.consecutive_frames:
            self.confidence_window.pop(0)
            self.action_window.pop(0)

        if (len(self.confidence_window) == self.consecutive_frames and
            all(c > self.confidence_threshold for c in self.confidence_window) and
            len(set(self.action_window)) == 1):
            self.model.initialize_cell_and_hidden_state()
            self.last_action = predicted_action
            self.last_action_time = time.time()
            return self.last_action, confidence.item()

        if len(self.hands_visible_window) == 5 and not any(self.hands_visible_window):
            self.model.initialize_cell_and_hidden_state()
            self.hands_visible_window = []

        return None, None

    def reset_hidden_state(self):
        self.model.initialize_cell_and_hidden_state()
