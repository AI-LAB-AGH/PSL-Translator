import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader

from models.model_transformer import TransformerModel
from models.model_LSTM import LSTMModel
from models.model_conv_LSTM import ConvLSTM
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from training import train, train_forecaster, display_results
from preprocessing.transforms import ExtractLandmarksWithRTMP, ExtractOpticalFlow
from preprocessing.datasets import JesterDataset, RTMPDataset, OFDataset


def separate_sample(model, test_loader, threshold=0.001):
    mse = torch.nn.MSELoss()
    model.initialize_cell_and_hidden_state()
    idx = random.randint(0, len(test_loader)-1)
    sample, _ = test_loader[idx]

    print(f'Sample index: {idx}. Cuts at:', end=' ')

    for frame in range(len(sample)-1):
        x = torch.tensor(sample[frame], dtype=torch.float32)
        x = x.view(1, 133, 2)
        outputs = model(x)
        outputs = outputs.view(outputs.shape[1] // 2, 2)
        tgt = torch.tensor(sample[frame+1], dtype=torch.float32)
        loss = mse(outputs, tgt)
        if loss.item() > threshold:
            print(frame+1, end=', ')


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
        
        # Extract landmarks
        x = transform([img])
        x = torch.tensor(x[0], dtype=torch.float32)

        # draw_landmarks(img, x)

        x = x.view(1, 133, 2)
        output = model(x)

        # Pass input through network
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        if confidence > 0.4:
            action_text = f'{predicted_action}'
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
            cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(confidence * 255), int(255 - confidence * 255)), 2, cv2.LINE_AA)
            cv2.imshow('Camera', img)
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        # Show image
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_real_time_inference_optical_flow(model, actions, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    model.initialize_cell_and_hidden_state()
    action_text = ""

    # Grab first frame so that there are 2 of them to process at the first inference step
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        return False
    cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', img)
    cv2.waitKey(1)

    prev = img

    while True:
        # Grab current frame
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            return False
        cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', img)

        curr = img

        # Extract optical flow
        flow = transform(prev, curr)

        # Pass input through network
        output = model(flow)
        output[0] = torch.nn.functional.softmax(output[0])
        confidence, predicted_index = torch.max(output, dim=1)
        predicted_action = actions[predicted_index.item()]

        # Output the recognized action
        if confidence > 0.6:
            action_text = f'{predicted_action}'
            print('\r'+ ' ' * 100, end='')
            print(f'\rRecognized action: {predicted_action} with confidence: {confidence.item():.2f}', end='')
        else:
            print('\r'+ ' ' * 100, end='')
            print(f'\rUnknown action. Most likely: {predicted_action} with confidence: {confidence.item():.2f}', end='')

        prev = curr

        # Show image
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
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
    label_map = None
    actions = None
    if os.path.isfile(labels):
        with open(labels, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    if label_map is not None:
        actions = np.array(list(label_map.keys()))
        num_classes = len(label_map)
    model_path = 'models/pretrained/'+model_type+'_'+dataset+'.pth'

    # Landmark extraction methods
    if model_type == 'LSTM' or model_type == 'Forecaster':
        extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')

    # Training params
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    if model_type == 'Forecaster':
        criterion = torch.nn.MSELoss
    else:
        criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    if model_type == 'LSTM' or model_type == 'Forecaster':
        transform = ExtractLandmarksWithRTMP(extractor)
    else:
        transform = None
    from_checkpoint = args.from_checkpoint
    
    # Model params
    input_shape = (29, (133 + 42) * 2) # All landmarks + 2 hands relative to the body source
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    match model_type:
        case 'Transformer':
            model = TransformerModel(input_shape[1], num_classes)

        case 'LSTM':
            model = LSTMModel(input_shape[1], hidden_size, num_layers, num_classes)

        case 'ConvLSTM':
            model = ConvLSTM(hidden_size, num_layers, num_classes)

        case 'Forecaster':
            model = LSTMModel(input_shape[1], hidden_size, num_layers, input_shape[1] - 42 * 2)

    if from_checkpoint:
        model.load_state_dict(torch.load(model_path))
    else:
        print('Loading training set...')
        match dataset:
            case 'jester':
                train_dataset = JesterDataset(root_dir_train, annotations_train, label_map, transform, None, 50)
                print('Done. Loading testing set...')
                test_dataset = JesterDataset(root_dir_test, annotations_test, label_map, transform, None, 10)

            case 'jester_RTMP':
                train_dataset = RTMPDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = RTMPDataset(root_dir_test, transform, None)

            case 'RGB_RTMP':
                train_dataset = RTMPDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = RTMPDataset(root_dir_test, transform, None)

            case 'RGB_OF':
                train_dataset = OFDataset(root_dir_train)
                print('Done. Loading testing set...')
                test_dataset = OFDataset(root_dir_test)

            case 'KSPJM_RTMP':
                train_dataset = RTMPDataset(root_dir_train)
                print('Done. Loading testing set...')
                test_dataset = RTMPDataset(root_dir_test)

        print('Done. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if model_type == 'Forecaster':
            results = train_forecaster(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        else:
            results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        display_results(results, actions)
    
    if model_type == 'LSTM':
        loader = RTMPDataset(root_dir_test)
    if model_type == 'ConvLSTM':
        run_real_time_inference_optical_flow(model, actions, transform=ExtractOpticalFlow())
    elif model_type == 'Forecaster':
        separate_sample(model, loader)
    else:
        run_real_time_inference(model, actions, transform)

if __name__ == "__main__":
    main()
