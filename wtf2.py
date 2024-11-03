import os

data_path = os.path.join('data', 'RGB', 'test')

for d in os.listdir(data_path):
    if d[-1] == '_':
        os.rename(os.path.join(data_path, d), os.path.join(data_path, d)[:len(os.path.join(data_path, d))-1])
