import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow
import os
import json
import numpy as np
from app.model_LSTM_transformer import LSTMTransformerModel
import torch
from app.helpers.gesture_recognition_handler import GestureRecognitionHandler
from preprocessing.landmark_extraction.rtmpose import RTMPoseDetector
from preprocessing.transforms import ExtractLandmarksWithRTMP

def load_stylesheet(app, file_path="app/assets/styles.qss"):
    with open(file_path, "r") as file:
        app.setStyleSheet(file.read())

if __name__ == "__main__":
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
    
    model = LSTMTransformerModel(128, 1, num_classes, 64)
    model_path = 'app/LSTM-Transformer_RGB_more_copy_2_RTMP_.pth'
    model.load_state_dict(torch.load(model_path))

    prediction_handler = GestureRecognitionHandler(model, label_map)

    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow(prediction_handler, transform)
    window.show()
    sys.exit(app.exec_())
