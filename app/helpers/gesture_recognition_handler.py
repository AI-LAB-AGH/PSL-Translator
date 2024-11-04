import torch
import cv2
import time

class GestureRecognitionHandler:
    def __init__(self, model, label_map, transform, confidence_threshold=0.95, window_width=60):
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

    def process_video(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access camera.")
            return None

        results = []
        while True:
            # Capture frame
            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
                break

            # Process frame
            action_text, confidence = self.process_frame(img)
            if action_text:
                results.append((action_text, confidence))
                
            if cv2.waitKey(1) == 27:  # Press 'Esc' to stop
                break

        cap.release()
        cv2.destroyAllWindows()
        return results

    def process_frame(self, frame):
        x = self.transform([frame])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)
        
        output = self.model(x)
        output = torch.nn.functional.softmax(output[0])
        confidences, predicted_indices = torch.topk(output, 3)
        
        confidence = confidences[0].item()
        predicted_action = self.actions[predicted_indices[0].item()]
        
        action_text = predicted_action if confidence > self.confidence_threshold else self.action_text
        
        self.tokens.append(action_text)
        decoded = self.dumb_decode(self.tokens)
        
        for token in decoded:
            self.tokens.remove(token)
        
        if decoded and decoded[-1] != self.prev:
            self.out.append(decoded[-1])
            self.prev = decoded[-1]

        if confidence > self.confidence_threshold:
            self.model.initialize_cell_and_hidden_state()

        print('output: ', self.out)
        print('decoded: ', decoded)
        return action_text, confidence, self.out

    def dumb_decode(self, sequence, window_width=15):
        previous_token = sequence[0]
        current_count = 1
        clean_sequence = []
        
        for idx, token in enumerate(sequence):
            if token != previous_token:
                if current_count >= window_width:
                    clean_sequence += sequence[idx - current_count + 1:idx + 1]
                current_count = 1
            else:
                current_count += 1
            previous_token = token
        if current_count >= window_width:
            clean_sequence += sequence[len(sequence) - current_count:]
        
        return self.ctc_decode(clean_sequence)

    def ctc_decode(self, sequence, blank_token='_'):
        decoded_output = []
        previous_token = None
        for token in sequence:
            if token != blank_token and token != previous_token:
                decoded_output.append(token)
            previous_token = token
        return decoded_output
