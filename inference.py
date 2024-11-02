import os
import cv2
import torch
import random
import mediapipe as mp
import matplotlib.pyplot as plt

def draw_landmarks(img, holistic) -> cv2.UMat:
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = holistic.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand, mp.solutions.holistic.HAND_CONNECTIONS)
    return img


def inference(model, label_map, transform):
    actions = dict([(value, key) for key, value in label_map.items()])
    window_width = 60
    tokens = ['' for _ in range(window_width)]
    threshold = 1.0
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    model.initialize_cell_and_hidden_state()
    action_text = ""

    while True:
        # Grab frame
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            return False
        
        # Extract landmarks
        x = transform([img])
        x = torch.tensor(x[0], dtype=torch.float32)
        x = x.view(1, 133, 2)
        output = model(x)

        # Pass input through network
        output = torch.nn.functional.softmax(output[0])
        confidences, predicted_indices = torch.topk(output, 3)
        # print(f'Predicted: {[(actions[index.item()], confidence.item()) for index, confidence in zip(predicted_indices, confidences)]}')
        confidence = confidences[0]
        predicted_action = actions[predicted_indices[0].item()]

        # Output the recognized action
        # --- Based on threshold ---
        action_text = f'{predicted_action}'
        # action_text = f'{predicted_action}' if confidence > threshold else action_text

        # --- Based on the mode of last `window_width` predictions
        tokens.append(action_text)
        tokens.pop(0)
        print(f'\r{[s for s in dumb_decode(tokens) if s != '']}')

        # --- Visualization ---
        # img = draw_landmarks(img, holistic)

        img = cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(confidence * 255), int(255 - confidence * 255)), 2, cv2.LINE_AA)

        if confidence > threshold:
            # --- Resetting LSTM states upon reaching threshold ---
            model.initialize_cell_and_hidden_state()
            print('\r'+ ' ' * 100, end='')
            print(f'Recognized action: {predicted_action} with confidence: {confidence.item():.2f}')
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'Unknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}')

        # Show image
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def ctc_decode(sequence, blank_token='_'):
    print(sequence)
    decoded_output = []
    
    previous_token = None
    for token in sequence:
        if token != blank_token and token != previous_token:
            decoded_output.append(token)
        previous_token = token
    
    return decoded_output


def dumb_decode(sequence, window_width=10):
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
    if current_count > len(sequence):
        clean_sequence = sequence
    
    return ctc_decode(clean_sequence)
