import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from preprocessing.rtmpose import RTMPoseDetector
from training import train, display_results
from preprocessing.utils import ExtractLandmarksWithRTMP
from dataloader import RTMPDataset
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

    return parser.parse_args()


def main():
    args = get_args()

    extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')
    model_type = args.model
    dataset = args.dataset
    from_checkpoint = args.from_checkpoint
    pretrained_path = 'models/pretrained/'+model_type+'_'+dataset+'.pth'
    
    root_dir_train = 'data/'+dataset+'/train'
    root_dir_test = 'data/'+dataset+'/test'
    labels = 'data/'+dataset+'/labels.json'
    label_map = None
    actions = None
    
    if os.path.isfile(labels):
        with open(labels, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    if label_map is not None:
        actions = np.array(list(label_map.keys()))

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    cut = args.cut
    skip_rate = args.skip_rate
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    
    match model_type:
        case _:
            model = None

    if from_checkpoint:
        print('Loading model from checkpoint...')
        model.load_state_dict(torch.load(pretrained_path))

    else:
        print('Loading training set...')
        match dataset:
            case 'RGB_RTMP':
                train_dataset = RTMPDataset(root_dir_train)
                print('Done. Loading testing set...')
                test_dataset = RTMPDataset(root_dir_test)

        print('Done. Starting training...')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = train(model, train_loader, test_loader, num_epochs, lr, cut, skip_rate, criterion, optimizer, pretrained_path)
        display_results(results, actions)
    
    # inference(model, label_map)

if __name__ == "__main__":
    main()
