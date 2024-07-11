import os
import cv2
import json
import torch
import keyboard
import numpy as np
import mediapipe as mp
from collections import deque
from torch.utils.data import DataLoader

from model import TransformerModel
from training import train, display_results
from dataloader import DistFromConsecTransform, DistFromFirstTransform, LandmarksDataset

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

def run_real_time_inference(model, actions, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

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
                height, width, _ = image.shape
                cv2.putText(image, "wykrywanie gestu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                keypoints.append(keypoint)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    cap.release()
                    cv2.destroyAllWindows()
                    return   
                                      
            input = transform()(keypoints).unsqueeze(0)
            output = model(input)
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

root_dir_train = 'data/landmarks/train'
root_dir_test = 'data/landmarks/test'
annotations_train = 'data/landmarks/annotations_train.csv'
annotations_test = 'data/landmarks/annotations_test.csv'
labels = 'labels_model_v1.json'
with open(labels, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
actions = np.array(list(label_map.keys()))
model_path = os.path.join('models', 'base_model_fixed_input_shape.pth')


def main():
    num_epochs = 50
    batch_size = 32
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    transform = DistFromFirstTransform
    save = False
    from_checkpoint = False
    
    input_shape = (29, 126)
    num_classes = len(label_map)
    model = TransformerModel(input_dim=input_shape[1], num_classes=num_classes)

    if from_checkpoint:
        model.load_state_dict(torch.load(model_path))
    else:
        print('Loading training set...')
        train_dataset = LandmarksDataset(root_dir_train, annotations_train, label_map, transform)
        print('Done. Loading testing set...')
        test_dataset = LandmarksDataset(root_dir_test, annotations_test, label_map, transform)
        print('Done. Starting training...')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, save)
        display_results(results, actions)
        
    run_real_time_inference(model, actions, transform)

if __name__ == "__main__":
    main()
