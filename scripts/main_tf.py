import numpy as np
import os
import cv2
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from tensorflow.keras.utils import register_keras_serializable
import keyboard
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, MultiHeadAttention
import tensorflow as tf

@register_keras_serializable(package='Custom')
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

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

def compute_differences(landmarks: np.array) -> np.array:
    differences = np.zeros((landmarks.shape[0]-1, landmarks.shape[1]))
    for frame in range(differences.shape[0]):
        differences[frame] = landmarks[frame+1] - differences[frame]
    return differences

PATH = os.path.join('data')
with open('labels_my_model_4.json', 'r') as f:
    label_map = json.load(f)
actions = np.array(list(label_map.keys()))
model = load_model('my_transformer_model.keras', custom_objects={'TransformerBlock': TransformerBlock})

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

def main():
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
                height,  width, _ = image.shape
                cv2.putText(image, "wykrywanie gestu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                keypoints.append(keypoint)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                        cap.release()
                        cv2.destroyAllWindows()
                        return   
                                      
            input = compute_differences(np.array(keypoints))
            prediction = model.predict(input[np.newaxis, :, :])
            keypoints = []  
            if np.amax(prediction) > 0.1:
                predicted_index = np.argmax(prediction)
                predicted_action = actions[predicted_index]
                action_text = f"{predicted_action}"
                print(f"Recognized action: {predicted_action} with confidence: {np.max(prediction):.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
