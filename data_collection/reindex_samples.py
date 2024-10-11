import os
import csv
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join('data', 'RGB'), help='Path to the root of data (RGB)')
    parser.add_argument('--first_index_to_move', type=int, help='First index of the chunk of data to move')
    parser.add_argument('--last_index_to_move', type=int, help='Last index of the chunk of samples to move')
    parser.add_argument('--target_first_index', type=int, help='Where to move first index')
    return parser.parse_args()


def main():
    # get CLI arguments
    args = get_args()
    data_path = args.data_path
    move_from = args.first_index_to_move
    last_idx = args.last_index_to_move
    move_to = args.target_first_index

    with open(os.path.join(data_path, 'annotations_train.csv')) as f:
        annotations_train = list(csv.reader(f))
    
    with open(os.path.join(data_path, 'annotations_test.csv')) as f:
        annotations_test = list(csv.reader(f))

    for i in range(last_idx - move_from + 1):
        ind = move_from + i
        if ind % 5 == 0:
            os.rename(os.path.join(data_path, 'test', str(ind)), os.path.join(data_path, 'test', str(move_to + i)))
            annotation_idx = next((i for i, record in enumerate(annotations_test) if int(record[0]) == ind), None)
            annotations_test[annotation_idx][0] = str(move_to + i)
        else:
            os.rename(os.path.join(data_path, 'train', str(ind)), os.path.join(data_path, 'train', str(move_to + i)))
            annotation_idx = next((i for i, record in enumerate(annotations_train) if int(record[0]) == ind), None)
            annotations_train[annotation_idx][0] = str(move_to + i)

    # sort annotations by index for good measure
    annotations_train.sort(key=lambda x : int(x[0]))
    annotations_test.sort(key=lambda x: int(x[0]))

    with open(os.path.join(data_path, 'annotations_train.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(annotations_train)
    
    with open(os.path.join(data_path, 'annotations_test.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(annotations_test)


if __name__ == '__main__':
    main()
