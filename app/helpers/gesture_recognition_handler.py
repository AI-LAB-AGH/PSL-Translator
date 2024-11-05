import torch
from translation.translator import Translator


class GestureRecognitionHandler:
    def __init__(self, model, label_map, transform, confidence_threshold=0.8, window_width=60):
        self.model = model
        self.actions = {value: key for key, value in label_map.items()}
        self.transform = transform
        self.threshold = confidence_threshold
        self.window_width = window_width
        
        self.model.initialize_cell_and_hidden_state()
        self.tokens = ['' for _ in range(window_width)]
        self.action_text = ""
        self.out = []
        self.prev = ''
        self.no_gesture_frames = 0
        self.decoded = []
        self.translation = ''

        self.translator = Translator()

    def process_frame(self, frame):
        x = self.transform([frame])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)
        
        output = self.model(x)
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidences, predicted_indices = torch.topk(output, 3)
        
        confidence = confidences[0].item()
        predicted_action = self.actions[predicted_indices[0].item()]

        self.action_text = f'{predicted_action}' if confidence > self.threshold else self.action_text

        # --- Based on the mode of last `window_width` predictions
        self.tokens.append(self.action_text)
        self.decoded = self.dumb_decode(self.tokens)
        for token in self.decoded:
            if token in self.tokens:
                self.tokens.remove(token)

        if self.decoded and self.decoded[-1] != self.prev:
            self.out.append(self.decoded[-1])
            self.prev = self.decoded[-1]
        
        if confidence > self.threshold:
            self.model.initialize_cell_and_hidden_state()
            print(f'{self.decoded}')
        
        if len(self.out) > 1 and self.out[-1] == 'blank':
            while 'blank' in self.out:
                self.out.remove('blank')
            self.translation = self.translator.translate(self.out)
            print(self.translation)
            #self.translation = ''
            self.out = []
            return self.translation, confidence, 'translation'
            
        if len(self.tokens) > self.window_width:
            self.tokens.pop(0)
        
        return self.action_text, confidence, 'gesture'

    def decode_and_translate(self):
        decoded = self.dumb_decode(self.tokens)
        self.tokens = []
        
        if decoded:
            translation = self.translate_gestures(decoded)
            return translation
        else:
            return ''

    def dumb_decode(self, sequence, window_width=15):
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

    def ctc_decode(self, sequence):
        decoded_output = []
        
        previous_token = None
        for token in sequence:
            if token != previous_token:
                decoded_output.append(token)
            previous_token = token
        
        return decoded_output

    def translate_gestures(self, gestures):
        translation = self.translator.translate(gestures)
        return translation
