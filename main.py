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
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from training import train, train_forecaster, display_results
from preprocessing.transforms import ExtractLandmarksWithRTMP, ExtractOpticalFlow
from preprocessing.datasets import JesterDataset, RTMPDataset, OFDataset
from inference import *


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
    parser.add_argument('--skip_rate', type=int, default=2, help='Skip rate for frames during training')
    parser.add_argument('--cut', type=int, default=0, help='Percentage of sample to be removed from the start and end')

    # LSTM hyperparameters
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=20, help='Hidden state dim in RNN model')

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
        num_classes = 219
    model_path = 'models/pretrained/'+model_type+'_'+dataset+'.pth'

    # Landmark extraction methods
    if model_type == 'LSTM' or model_type == 'Forecaster':
        extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')

    # Training params
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    cut = args.cut
    skip_rate = args.skip_rate
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    if model_type == 'LSTM' or model_type == 'Forecaster':
        transform = ExtractLandmarksWithRTMP(extractor)
    else:
        transform = None
    from_checkpoint = args.from_checkpoint
    
    # Model params
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    match model_type:
        # case 'Transformer':
        #     model = TransformerModel(input_size, num_classes)

        case 'LSTM':
            model = LSTMModel(hidden_size, num_layers, num_classes)

        case 'ConvLSTM':
            model = ConvLSTM(hidden_size, num_layers, num_classes)

        # case 'Forecaster':
        #     model = LSTMModel(hidden_size, num_layers, input_size - 21*2*2)

    if from_checkpoint:
        print('Loading model from checkpoint...')
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
            results = train(model, train_loader, test_loader, num_epochs, lr, cut, skip_rate, criterion, optimizer, model_path)
        display_results(results, actions)
    
    inference(model, label_map, transform)

if __name__ == "__main__":
    main()
