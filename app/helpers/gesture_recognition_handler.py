import torch
import cv2

class GestureRecognitionHandler:
    def __init__(self, model, label_map, transform, confidence_threshold=0.9, window_width=60):
        self.model = model
        self.actions = {value: key for key, value in label_map.items()}
        self.transform = transform
        self.confidence_threshold = confidence_threshold
        self.window_width = window_width
        self.tokens = []
        
        self.model.initialize_cell_and_hidden_state()
        self.action_text = ""
        self.out = []
        self.prev = ''
        self.no_gesture_frames = 0

    def process_frame(self, frame):
        x = self.transform([frame])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)
        
        output = self.model(x)
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidences, predicted_indices = torch.topk(output, 3)
        
        confidence = confidences[0].item()
        predicted_action = self.actions[predicted_indices[0].item()]
        
        if confidence > self.confidence_threshold:
            action_text = predicted_action
            self.no_gesture_frames = 0 
            self.tokens.append(action_text)
            if action_text != self.prev:
                self.out.append(action_text)
                self.prev = action_text
        else:
            action_text = ''
            self.no_gesture_frames += 1
            self.tokens.append('_')
            
        if len(self.tokens) > self.window_width:
            self.tokens.pop(0)

        if self.no_gesture_frames >= 10:
            translation = self.decode_and_translate()
            self.no_gesture_frames = 0 
            self.out = []
            return translation, confidence, 'translation'
        
        if confidence > self.confidence_threshold:
            self.model.initialize_cell_and_hidden_state()
        
        return action_text, confidence, 'gesture'

    def decode_and_translate(self):
        decoded = self.dumb_decode(self.tokens)
        self.tokens = []
        
        if decoded:
            translation = self.translate_gestures(decoded)
            return translation
        else:
            return ''

    def dumb_decode(self, sequence, window_width=3):
        previous_token = sequence[0]
        current_count = 1
        clean_sequence = []
        
        for idx, token in enumerate(sequence):
            if token != previous_token:
                if current_count >= window_width:
                    clean_sequence += sequence[idx-current_count+1:idx+1]
                current_count = 1
            else:
                current_count += 1
            previous_token = token
        if current_count >= window_width:
            clean_sequence += sequence[idx-current_count+1:idx+1]
        
        return self.ctc_decode(clean_sequence)

    def ctc_decode(self, sequence, blank_token='_'):
        decoded_output = []
        
        previous_token = None
        for token in sequence:
            if token != blank_token and token != previous_token:
                decoded_output.append(token)
            previous_token = token
        
        return decoded_output

    def translate_gestures(self, gestures):
        # TODO: add a proper translation
        translation = ' '.join(gestures)
        return translation
