import os
import csv
from distutils.dir_util import copy_tree

def reorganize_and_annotate(root_dir: str, target_train_dir: str, target_test_dir: str, annotations: dict, target_train_csv: str, target_test_csv: str):
    sample_count = 0

    if not os.path.isdir(target_train_dir):
        os.makedirs(target_train_dir)
    if not os.path.isdir(target_test_dir):
        os.makedirs(target_test_dir)

    with open(target_train_csv, 'a', newline='', encoding='utf-8') as train:
        with open(target_test_csv, 'a', newline='', encoding='utf-8') as test:
            writer_train = csv.writer(train, delimiter=',')
            writer_test = csv.writer(test, delimiter=',')

            for seq in os.listdir(root_dir): # 0, 1, 2, ...
                label = annotations[seq]
                src = os.path.join(root_dir, seq)

                if sample_count % 5 == 0:
                    dst = os.path.join(target_test_dir, seq)
                    writer_test.writerow([seq, label])
                else:
                    dst = os.path.join(target_train_dir, seq)
                    writer_train.writerow([seq, label])
                    
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                copy_tree(src, dst)
                sample_count += 1

    train.close()
    test.close()

def main():
    # ------------ INSTRUCTIONS ------------
    # After extracting jester.zip to some directory 'dir/', run this script from inside 'dir/'.
    # Before running the script, copy the 'annotations.csv' file and place it inside 'dir/'. Also, please move the 'dir/labels/labels.csv' file into 'dir/'
    # The script will run for a few minutes, reorganizing the files. After it's done, you can delete the '20bn-jester-v1/' and 'labels/' directories.

    root_dir = '20bn-jester-v1'
    target_train_dir = 'train'
    target_test_dir = 'test'
    target_train_csv = 'annotations_train.csv'
    target_test_csv = 'annotations_test.csv'
    with open('annotations.csv', mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        annotations = {row[0]: row[1] for row in reader}

    reorganize_and_annotate(root_dir, target_train_dir, target_test_dir, annotations, target_train_csv, target_test_csv)

if __name__ == '__main__':
    main()