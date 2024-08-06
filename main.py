import cv2
import json
import torch
# import keyboard
from pynput import keyboard
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
from torchvision import transforms

from model_transformer import TransformerModel
from model_LSTM import LSTMModel
from training import train, display_results
from data_prep.transforms import ExtractLandmarks, ComputeDistSource, ComputeDistFirst, ComputeDistConsec, ComputeDistNetNoMovement, ComputeDistNetWithMovement
from data_prep.datasets import LandmarksDataset, JesterDataset, ProcessedDataset


def draw_landmarks(img, holistic):
    results = holistic.process(img)
    
    mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    
    return img


def run_real_time_inference(model, actions, holistic, transform):
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
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', img)
        
        # Process frame to obtain model input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left, right = transform([img])
        left = left = left[0].view(1, 1, -1)
        right = right[0].view(1, 1, -1)

        # Pass input through network
        output = model(left, right)
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        if confidence > 0.4:
            action_text = f'{predicted_action}'
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        # Draw detected landmarks and show image
        # img = draw_landmarks(img, holistic)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_set_size_inference(model, actions, holistic, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    action_text = "" 
    while cap.isOpened():
        success, img = cap.read()
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', img)
        
        # pynput setup for key listening
        space_pressed = False
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while not keyboard.is_pressed('space'):
            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
                return False
            
            img = draw_landmarks(img, holistic)
            height, width, _ = img.shape
            cv2.putText(img, action_text, (width // 2 - len(action_text) * 10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "nacisnij SPACJE by rozpoznac gest", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Camera', img)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return
        
        listener.stop()
        
        frames = []
        print("Recognizing the action....") 
        while len(frames) < 30:
            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
                return True
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = draw_landmarks(img, holistic)
            height, width, _ = img.shape
            cv2.putText(img, "wykrywanie gestu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera', img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return   
                
        output = None
        model.initialize_cell_and_hidden_state()
        
        lefts, rights = transform(frames)
        for frame in range(len(lefts)):
            left = lefts[frame].view(1, 1, -1)
            right = rights[frame].view(1, 1, -1)
            output = model(left, right)

        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]
        frames = []

        if confidence > 0.4:
            action_text = f"{predicted_action}"
            print(f"Recognized action: {predicted_action} with confidence: {confidence.item():.2f}")
        else:
            print(f"Unknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


model_type = 'lstm'
dataset = 'landmarks_P'
root_dir_train = 'data/'+dataset+'/train'
root_dir_test = 'data/'+dataset+'/test'
annotations_train = 'data/'+dataset+'/annotations_train.csv'
annotations_test = 'data/'+dataset+'/annotations_test.csv'
labels = 'data/'+dataset+'/labels.json'
with open(labels, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
actions = np.array(list(label_map.keys()))
model_path = 'models/'+model_type+'_model.pth'


def main():
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    num_epochs = 10
    batch_size = 1
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    transform = transforms.Compose([ExtractLandmarks(holistic),
                                    ComputeDistNetWithMovement()])
    from_checkpoint = False
    
    input_shape = (29, 21*3)
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
            case 'landmarks':
                train_dataset = LandmarksDataset(root_dir_train, annotations_train, label_map, transform)
                print('Done. Loading testing set...')
                test_dataset = LandmarksDataset(root_dir_test, annotations_test, label_map, transform)
                
            case 'jester':
                train_dataset = JesterDataset(root_dir_train, annotations_train, label_map, transform, None, 50)
                print('Done. Loading testing set...')
                test_dataset = JesterDataset(root_dir_test, annotations_test, label_map, transform, None, 10)

            case 'RGB_P':
                train_dataset = ProcessedDataset(root_dir_train, transform, None, -1)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None, -1)

            case 'landmarks_P':
                train_dataset = ProcessedDataset(root_dir_train, transform, None, -1)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None, -1)

        print('Done. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        display_results(results, actions)
        
    run_real_time_inference(model, actions, holistic, transform)

if __name__ == "__main__":
    main()
