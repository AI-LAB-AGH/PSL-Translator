import re
from translation.translation_dataloader import TranslationDataset
import random

def data_cleaning():
    with open("third_data.txt", "r") as i:
        with open("dataset_part2.txt", "w+") as o:
            for line in i:
                if line.strip():
                    o.write(re.sub(r"[.,!?]", "", line))

def data_check():
    data = TranslationDataset("dataset_part2.txt")
    for i in range(50):
        num = random.randint(0, len(data))
        print(data[num])

def data_concatenate():
    with open("dataset_part1.txt", "a+") as f:
        for line in open("data.txt", "r"):
            f.write(line)
        for line in open("second_data_cleaned.txt", "r"):
            f.write(line)



data_check()