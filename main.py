import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from models.model_transformer import TransformerModel
from models.model_LSTM import LSTMModel
from models.model_forecaster import Forecaster
from models.model_conv_LSTM import ConvLSTM
from models.model_LSTM_transformer import LSTMTransformerModel
from models.model_LSTM_transformer_2 import LSTMTransformerModel2
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from training import train, train_forecaster, display_results
from preprocessing.transforms import ExtractLandmarksWithRTMP, ExtractOpticalFlow
from preprocessing.datasets import JesterDataset, RTMPDataset, OFDataset
from inference import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model to use (LSTM, ConvLSTM, Transformer, LSTM-Transformer)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset (one of those in data/ directory) suitable for the chosen model')

    # flags
    parser.add_argument('--from_checkpoint', type=bool, default=False, help="Flag whether to train the model or load an already trained one")

    # model hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')

    # LSTM and Transformer hyperparameters
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden state dim in RNN model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in Transformer')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of Transformer encoder layers')

    return parser.parse_args()


def main():
    args = get_args()

    model_type = args.model
    dataset = args.dataset
    root_dir_train = 'data/preprocessed/' + dataset + '/train'
    root_dir_test = 'data/preprocessed/' + dataset + '/test'
    annotations_train = 'data/preprocessed/' + dataset + '/annotations_train.csv'
    annotations_test = 'data/preprocessed/' + dataset + '/annotations_test.csv'
    labels = 'data/preprocessed/' + dataset + '/labels.json'
    print('labels:', labels)
    print('root_dir_train:', root_dir_train)
    label_map = None
    actions = None
    if os.path.isfile(labels):
        print('Loading labels...')
        with open(labels, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

    if label_map is not None:
        actions = np.array(list(label_map.keys()))
        num_classes = 219
    # model_path = 'models/pretrained/' + model_type + '_' + dataset + '_' + '.pth'
    model_path = 'models/pretrained/' + model_type + '_' + '7_12' + '_' + '.pth'

    # Landmark extraction methods
    if model_type in ['LSTM', 'Forecaster', 'LSTM-Transformer']:
        extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')

    # Training params
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    if model_type in ['LSTM', 'Forecaster', 'LSTM-Transformer']:
        transform = ExtractLandmarksWithRTMP(extractor)
    else:
        transform = None
    from_checkpoint = args.from_checkpoint

    # Model params
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_heads = args.num_heads  # Number of heads for Transformer
    transformer_layers = args.transformer_layers  # Number of Transformer layers
    attention_dim = 64
    
    # Model initialization
    match model_type:
        case 'Transformer':
            input_size = 0
            model = TransformerModel(input_size, num_classes)

        case 'LSTM':
            model = LSTMModel(hidden_size, num_layers, num_classes)

        case 'ConvLSTM':
            model = ConvLSTM(hidden_size, num_layers, num_classes)

        case 'LSTM-Transformer':
            model = LSTMTransformerModel(hidden_size, num_layers, num_classes, attention_dim)
        
        case 'LSTM-Transformer-2':
            model = LSTMTransformerModel2(hidden_size, num_layers, num_classes, attention_dim)
            
        case 'Forecaster':
            model = LSTMModel(hidden_size, num_layers, input_size - 21*2*2)

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

            case dataset if dataset.startswith('RGB_more_') and dataset.endswith('_RTMP'):
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
            
            case _:
                train_dataset = RTMPDataset(root_dir_train, transform, None)
                print('Done. Loading testing set...')
                test_dataset = RTMPDataset(root_dir_test, transform, None)

        print('Done. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if model_type == 'Forecaster':
            results = train_forecaster(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        else:
            results = train(model, train_loader, test_loader, num_epochs, lr, criterion, optimizer, model_path)
        display_results(results, actions)
        results_path = model_type + '_' + '7_12' + '_' + '.json'
        with open(results_path, 'w') as f:
            json.dump(results, f)

    if model_type == 'ConvLSTM':
        inference_optical_flow(model, actions, transform=ExtractOpticalFlow())
    elif model_type == 'Forecaster':
        loader = RTMPDataset(root_dir_test)
        separate_sample(model, transform, loader)
    else:
        inference(model, label_map, transform)


if __name__ == "__main__":
    main()
