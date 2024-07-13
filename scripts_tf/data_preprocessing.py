import os
import numpy as np

def create_directory(action):
    if not os.path.exists(os.path.join("data_updated/", action)):
        os.makedirs(os.path.join("data_updated/", action))

    sequence_list = os.listdir(os.path.join("data_updated/", action))
    sequences_no = len(sequence_list)

    os.makedirs(os.path.join("data_updated/", action, str(sequences_no)), exist_ok=True)

def compute_and_save_differences(landmarks: np.array, path: str) -> None:
    differences = np.zeros((landmarks.shape[0]-1, landmarks.shape[1]))
    for frame in range(differences.shape[0]):
        frame_path = os.path.join(path, str(frame))
        differences[frame] = landmarks[frame] - landmarks[frame+1]
        np.save(frame_path, differences[frame])

def main():
    PATH = os.path.join('data')
    PATH_U = os.path.join('data_updated')

    for action in os.listdir(PATH): # BARDZO_DOBRZE, ...
        action_path = os.path.join(PATH, action)
        action_path_u = os.path.join(PATH_U, action)
        if os.path.isdir(action_path_u):
            continue
        
        for seq in sorted(os.listdir(action_path), key=lambda a: int(a)): # 0, 1, 2, ...
            temp = []
            seq_path = os.path.join(action_path, seq)
            seq_path_u = os.path.join(action_path_u, seq)
            create_directory(action)
            
            for frame in sorted(os.listdir(seq_path), key=lambda a: int(os.path.splitext(a)[0])): # 0, 1, 2, ...
                npy_path = os.path.join(seq_path, frame)
                if os.path.isfile(npy_path):
                    npy = np.load(npy_path)
                    temp.append(npy)
            
            if temp:
                temp = np.array(temp)
                compute_and_save_differences(temp, seq_path_u)
                print(f'Action {action}, sequence {seq} complete')

if __name__ == '__main__':
    main()
