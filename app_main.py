import sys
import argparse
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow
import os
import json
import numpy as np
from app.model_LSTM_transformer import LSTMTransformerModel
from models.model_LSTM import LSTMModel
import torch
from app.helpers.gesture_recognition_handler import GestureRecognitionHandler
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from preprocessing.transforms import ExtractLandmarksWithRTMP


def load_stylesheet(app, file_path="app/assets/styles.qss"):
    with open(file_path, "r") as file:
        app.setStyleSheet(file.read())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='LSTM',
                        help='Model type to use (LSTM, ConvLSTM, Transformer, LSTM-Transformer)')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the model weights file')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden state dimension in the model')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers in the model')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of gesture classes (optional)')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_type = args.model_type
    model_path = args.model_path
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    
    extractor = RTMPoseDetector('preprocessing/landmark_extraction/end2end.onnx')
    transform = ExtractLandmarksWithRTMP(extractor)
    
    labels = 'app/labels.json'
    label_map = None
    if os.path.isfile(labels):
        with open(labels, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

    if label_map is not None:
        actions = np.array(list(label_map.keys()))
        num_classes = 219
    
    if model_type == 'LSTM':
        model = LSTMModel(hidden_size, num_layers, num_classes)
    elif model_type == 'LSTM-Transformer':
        model = LSTMTransformerModel(hidden_size, num_layers, num_classes, 64)
        
    model.load_state_dict(torch.load(model_path))

    prediction_handler = GestureRecognitionHandler(model, label_map, transform)

    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow(prediction_handler, transform, model)
    window.show()
    sys.exit(app.exec_())
