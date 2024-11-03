import os

data_path = os.path.join('data', 'RGB', 'train')

dirs = [int(d) for d in os.listdir(data_path)]
dirs.sort()

i = 1488
for d in dirs:
    if d > 833:
        os.rename(os.path.join(data_path, str(d)), os.path.join(data_path, str(i) + '_'))
        if (i + 1) % 5 == 0:
            i += 2
        else:
            i += 1

# data_path = os.path.join('data', 'RGB', 'test')

# dirs = [int(d) for d in os.listdir(data_path)]
# dirs.sort()

# i = 1490
# for d in dirs:
#     if d > 833:
#         os.rename(os.path.join(data_path, str(d)), os.path.join(data_path, str(i) + '_'))
#         i += 5
