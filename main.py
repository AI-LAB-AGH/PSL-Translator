import cv2
import json
import torch
import argparse
import numpy as np
import mediapipe as mp
from pynput import keyboard
from torchvision import transforms
from torch.utils.data import DataLoader

from models.model_transformer import TransformerModel
from models.model_LSTM import LSTMModel
from models.model_conv_LSTM import ConvLSTM
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from training import train, display_results
from preprocessing.transforms import ExtractLandmarksWithMP, ExtractLandmarksWithRTMP
from preprocessing.datasets import LandmarksDataset, JesterDataset, ProcessedDataset, OFDataset


def draw_landmarks(img, landmarks):
    h, w, c = img.shape

    source_body = landmarks[0].clone()
    source_face = landmarks[53].clone()
    source_left = landmarks[100].clone()
    source_right = landmarks[121].clone()

    body = landmarks[:17]
    face = landmarks[23:91]
    left = landmarks[91:112]
    right = landmarks[112:]

    # for l in body:
    #     cv2.line(img, (int(source_body[0] * w), int(source_body[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 0, 255))

    # for l in face:
    #     cv2.line(img, (int(source_face[0] * w), int(source_face[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 0, 255))

    for l in left:
        # cv2.line(img, (int(source_left[0] * w), int(source_left[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 0, 255))
        cv2.line(img, (int(source_body[0] * w), int(source_body[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 255, 0))

    for l in right:
        # cv2.line(img, (int(source_right[0] * w), int(source_right[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 0, 255))
        cv2.line(img, (int(source_body[0] * w), int(source_body[1] * h)), (int(l[0] * w), int(l[1] * h)), (0, 255, 0))

    return img


def run_real_time_inference(model, actions, transform):
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
        x = transform([img])
        x = torch.tensor(x[0], dtype=torch.float32)

        # draw_landmarks(img, x)

        x = x.view(1, 1, 133, 2)
        output = model(x)

        # Pass input through network
        output[0] = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        if confidence > 0.0:
            action_text = f'{predicted_action}'
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        # Show image
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == 27:
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model to use (LSTM, ConvLSTM, Transformer)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset (one of those in data/ directory) suitable for the chosen model')

    # flags
    parser.add_argument('--from_checkpoint', type=bool, default=False, help="Flag whether to train the model or load an already trained one")

    # model hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')

    # LSTM hyperparameters
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=120, help='Hidden state dim in RNN model')

    return parser.parse_args()


def main():
    args = get_args()

    model_type = args.model
    dataset = args.dataset
    root_dir_train = 'data/'+dataset+'/train'
    root_dir_test = 'data/'+dataset+'/test'
    annotations_train = 'data/'+dataset+'/annotations_train.csv'
    annotations_test = 'data/'+dataset+'/annotations_test.csv'
    labels = 'data/'+dataset+'/labels.json'
    with open(labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    actions = np.array(list(label_map.keys()))
    model_path = 'models/pretrained/'+model_type+'_'+dataset+'.pth'

    # Landmark extraction methods
    # holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')

    # Training params
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    transform = transforms.Compose([ExtractLandmarksWithRTMP(extractor)])
    from_checkpoint = args.from_checkpoint
    print(from_checkpoint)
    
    # Model params
    input_shape = (29, 133 * 2)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = len(label_map)

    match model_type:
        case 'Transformer':
            model = TransformerModel(input_shape[1], num_classes)

        case 'LSTM':
            model = LSTMModel(input_shape[1], hidden_size, num_layers, num_classes)

        case 'ConvLSTM':
            model = ConvLSTM(hidden_size, num_layers, num_classes)

    if from_checkpoint:
        model.load_state_dict(torch.load(model_path))
    else:
        print('Loading training set...')
        match dataset:
            case 'landmarks':
                train_dataset = LandmarksDataset(root_dir_train, annotations_train, label_map, transform)
                print('Done. Loading testing set...')
                test_dataset = LandmarksDataset(root_dir_test, annotations_test, label_map, transform)
                
            case 'landmarks_P':
                train_dataset = ProcessedDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None)

            case 'jester':
                train_dataset = JesterDataset(root_dir_train, annotations_train, label_map, transform, None, 50)
                print('Done. Loading testing set...')
                test_dataset = JesterDataset(root_dir_test, annotations_test, label_map, transform, None, 10)

            case 'jester_P':
                train_dataset = ProcessedDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None)

            case 'jester_RTMP':
                train_dataset = ProcessedDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None)

            case 'RGB_P':
                train_dataset = ProcessedDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None)

            case 'RGB_RTMP':
                train_dataset = ProcessedDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = ProcessedDataset(root_dir_test, transform, None)

            case 'RGB_OF':
                train_dataset = OFDataset(root_dir_train)
                print('Done. Loading testing set...')
                test_dataset = OFDataset(root_dir_test)


        print('Done. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        display_results(results, actions)
        
    run_real_time_inference(model, actions, transform)

if __name__ == "__main__":
    main()
