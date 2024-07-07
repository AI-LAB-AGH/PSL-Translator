import os
import cv2
import json
import torch
import keyboard
import numpy as np
import mediapipe as mp
from collections import deque
from training.transformers_pytorch import TransformerModel

def process_image_and_extract_keypoints(cap, holistic):
    success, image = cap.read()
    if not success:
        print("Failed to capture image.")
        return False

    image = cv2.flip(image, 1)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    
    keypoints = extract_keypoints(results)
    
    return image, keypoints

def extract_keypoints(results):
    keypoints = np.array([])
    for landmark_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if landmark_list is not None:
            for landmark in landmark_list.landmark:
                keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
        else:
            keypoints = np.append(keypoints, np.zeros(21*3))
            
    return keypoints


PATH = os.path.join('models', 'base_model_fixed_input_shape.pth')

with open('labels_model_v1.json', 'r') as f:
    label_map = json.load(f)
    
actions = np.array(list(label_map.keys()))

input_shape = (30, 126)
num_classes = len(actions)
model = TransformerModel(input_dim=input_shape[1], num_classes=num_classes)
model.load_state_dict(torch.load(PATH))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

def main():
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        action_text = "" 
        while cap.isOpened():
            image, keypoint = process_image_and_extract_keypoints(cap, holistic)
            cv2.putText(image, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Camera', image)
            
            print("Press SPACE to recognize the action.")
            while not keyboard.is_pressed('space'):
                image, keypoint = process_image_and_extract_keypoints(cap, holistic)
                height, width, _ = image.shape
                cv2.putText(image, action_text, (width // 2 - len(action_text) * 10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "nacisnij SPACJE by rozpoznac gest", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            keypoints = deque(maxlen=30)
            print("Recognizing the action....") 
            while len(keypoints) < 30:
                image, keypoint = process_image_and_extract_keypoints(cap, holistic)
                height,  width, _ = image.shape
                cv2.putText(image, "wykrywanie gestu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                keypoints.append(keypoint)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    cap.release()
                    cv2.destroyAllWindows()
                    return   
                                      
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
            output = model(keypoints_tensor)
            prob, predicted_index = torch.max(output, dim=1)

            predicted_action = actions[predicted_index.item()]
            keypoints = []
            if prob > 0.1:
                action_text = f"{predicted_action}"
                print(f"Recognized action: {predicted_action} with confidence: {prob.item():.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
