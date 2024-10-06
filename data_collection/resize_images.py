import cv2
import os


# Modify the paths accordingly
data_path = os.path.join('data', 'RGB')
resized_path = os.path.join('data', 'resized')

def resize(subset: str):
    '''
    subset is eigher "train" or "test"
    '''
    for dir in os.listdir(os.path.join(data_path, subset)):
        sample_path = os.path.join(data_path, subset, dir)
        resized_sample_path = os.path.join(resized_path, subset, dir)
        
        if not os.path.exists(resized_sample_path):
            os.mkdir(resized_sample_path)
        
        for filename in os.listdir(sample_path):
            img = cv2.imread(os.path.join(sample_path, filename))
            img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(resized_sample_path, filename), img)


resize('train')
resize('test')
