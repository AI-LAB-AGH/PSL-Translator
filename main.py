import os
import cv2
import json
import torch
import keyboard
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
from torchvision import transforms

from model_transformer import TransformerModel
from model_LSTM import LSTMModel
from training import train, display_results
from dataloader import ComputeDistSource, ComputeDistFirst, LandmarksDataset, JesterDataset

def draw_landmarks(img, holistic):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    
    return img, results

def extract_keypoints(results):
    keypoints = np.array([])
    for landmark_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if landmark_list is not None:
            for landmark in landmark_list.landmark:
                keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
        else:
            keypoints = np.append(keypoints, np.zeros(21*3))
    return keypoints

def run_real_time_inference(model, actions, holistic):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    action_text = "" 
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', img)
        
        while not keyboard.is_pressed('space'):
            success, img = cap.read()
            img = cv2.flip(img, 1)
            if not success:
                print("Failed to capture image.")
                return False
            
            img, results = draw_landmarks(img, holistic)
            height, width, _ = img.shape
            cv2.putText(img, action_text, (width // 2 - len(action_text) * 10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "nacisnij SPACJE by rozpoznac gest", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Camera', img)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return
            
        frames = []
        keypoints = []
        print("Recognizing the action....") 
        while len(frames) < 30:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            if not success:
                print("Failed to capture image.")
                return False
            
            img, results = draw_landmarks(img, holistic)
            height, width, _ = img.shape
            cv2.putText(img, "wykrywanie gestu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera', img)
            frames.append(img)
            keypoints.append(extract_keypoints(results))

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return   
                                    
        keypoints_array = np.array(keypoints)
        input_tensor = torch.tensor(keypoints_array).float().unsqueeze(0)  # Ensure tensor and add batch dimension
        output = model(input_tensor)
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]
        frames = []

        if confidence > 0.2:
            action_text = f"{predicted_action}"
            print(f"Recognized action: {predicted_action} with confidence: {confidence.item():.2f}")
        else:
            print(f"Unknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model_type = 'transformer'
# dataset = 'jester/jester/jester'
dataset = 'landmarks/landmarks'
root_dir_train = 'data/'+dataset+'/train'
root_dir_test = 'data/'+dataset+'/test'
annotations_train = 'data/'+dataset+'/annotations_train.csv'
annotations_test = 'data/'+dataset+'/annotations_test.csv'
labels = 'data/'+dataset+'/labels.json'
with open(labels, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
actions = np.array(list(label_map.keys()))
model_path = 'models/'+model_type+'_model_without_transforms.pth'

def main():
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    num_epochs = 20
    batch_size = 1
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    from_checkpoint = True
    
    input_shape = (29, 126)
    hidden_size = 20
    num_layers = 1
    num_classes = len(label_map)

    match model_type:
        case 'transformer':
            model = TransformerModel(input_shape[1], num_classes)

        case 'lstm':
            model = LSTMModel(input_shape[1], hidden_size, num_layers, num_classes)

    if from_checkpoint:
        model.load_state_dict(torch.load(model_path))
    else:
        print('Loading training set...')
        match dataset:
            case 'landmarks/landmarks':
                train_dataset = LandmarksDataset(root_dir_train, annotations_train, label_map, None)
                print('\nDone. Loading testing set...')
                test_dataset = LandmarksDataset(root_dir_test, annotations_test, label_map, None)
                
            case 'jester/jester/jester':
                train_dataset = JesterDataset(root_dir_train, annotations_train, label_map, None, 50)
                print('\nDone. Loading testing set...')
                test_dataset = JesterDataset(root_dir_test, annotations_test, label_map, None, 10)

        print('\nDone. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        display_results(results, actions)
        
    run_real_time_inference(model, actions, holistic)

if __name__ == "__main__":
    main()
