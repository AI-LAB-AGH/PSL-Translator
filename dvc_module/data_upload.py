import os
import csv
import shutil
from azure.storage.blob import BlobServiceClient

from config import CONNECT, CONTAINER

class DataUploader:
    def __init__(self, dataset_path='data/test', tmp_path='data/test'):
        self.dataset_path = dataset_path
        self.tmp_path = tmp_path

        print('INFO: Establishing Azure connection')
        blob_service_client = BlobServiceClient.from_connection_string(CONNECT)
        self.container_client = blob_service_client.get_container_client(CONTAINER)
        print('INFO: Connection established')

        self.sample_count = self.count_samples()


    def count_samples(self):
        local_count = len(os.listdir(os.path.join(self.dataset_path, 'train'))) + len(os.listdir(os.path.join(self.dataset_path, 'test')))
        remote_count = 0

        for blob in self.container_client.walk_blobs(name_starts_with="train/", delimiter='/'):
            if blob.name.endswith('/'):
                remote_count += 1
        for blob in self.container_client.walk_blobs(name_starts_with="test/", delimiter='/'):
            if blob.name.endswith('/'):
                remote_count += 1

        if local_count == remote_count:
            print('INFO: Everything up to date')
        else:
            print(f'WARNING: Remote ahead by {remote_count - local_count} samples. Pull will be carried out automatically before pushing.')

        return remote_count


    def fetch_annotations(self, subset):
        blob_client = self.container_client.get_blob_client(f"annotations_{subset}.csv")
        download_blob = blob_client.download_blob()
        downloaded_data = download_blob.readall()
        with open(os.path.join(self.dataset_path, f"annotations_{subset}.csv"), "wb") as file:
            file.write(downloaded_data)


    def pull(self, subset):
        self.fetch_annotations(subset)
        print(f'PULLING: Fetched {subset} annotations')

        with open(os.path.join(self.dataset_path, f'annotations_{subset}.csv'), mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            annotations = {row[0]: row[1] for row in reader}

        for key in annotations:
            if not os.path.isdir(os.path.join(self.dataset_path, subset, key)):
                prefix = f'{subset}/{key}/'
                os.makedirs(os.path.join(self.dataset_path, prefix), exist_ok=True)
                for blob in self.container_client.walk_blobs(name_starts_with=prefix):
                    blob_client = self.container_client.get_blob_client(blob.name)
                    download_blob = blob_client.download_blob()
                    downloaded_data = download_blob.readall()
                    with open(os.path.join(self.dataset_path, subset, key, blob.name.replace(prefix, '')), "wb") as file:
                        file.write(downloaded_data)
                print(f'PULLING: Pulling {subset} sample no. {key}')
        print(f'PULLING: Pulled all {subset} samples')


    def upload_sample(self, root_dir, label):
        if self.sample_count % 5 == 0:
            subset = 'test'
        else:
            subset = 'train'
        
        for frame in os.listdir(root_dir):
            with open(os.path.join(root_dir, frame), "rb") as data:
                blob_client = self.container_client.get_blob_client(f'{subset}/{self.sample_count}/{frame}')
                blob_client.upload_blob(data)

        if subset == 'train':
            with open(f'{self.dataset_path}/annotations_train.csv', "a") as f:
                f.write(f'{self.sample_count},{label}\n')
        else:
            with open(f'{self.dataset_path}/annotations_test.csv', "a") as f:
                f.write(f'{self.sample_count},{label}\n')

        print(F'PUSHING: Pushing {subset} sample {self.sample_count} with label {label}')
        self.sample_count += 1


    def update_annotations(self, subset):
        blob_client = self.container_client.get_blob_client(f"annotations_{subset}.csv")
        with open(f'{self.dataset_path}/annotations_{subset}.csv', "rb") as data:
            blob_client = self.container_client.get_blob_client(f"annotations_{subset}.csv")
            blob_client.upload_blob(data, overwrite=True)
        print(f'PUSHING: Updated {subset} annotations')


    def move_sample(self, sample):
        if (self.sample_count-1) % 5 == 0:
            subset = 'test'
        else:
            subset = 'train'

        src = os.path.join(self.tmp_path, sample)
        dest = os.path.join(self.dataset_path, subset, str(self.sample_count-1))
        os.makedirs(dest, exist_ok=True)

        for filename in os.listdir(src):
            src_file = os.path.join(src, filename)
            dest_file = os.path.join(dest, filename)
            shutil.move(src_file, dest_file)
        os.rmdir(src)


    def push(self, src):
        self.pull('train')
        self.pull('test')

        if not os.path.exists(os.path.join(src, "annotations.csv")):
            print('PUSHING: Nothing to push')
            return
        
        with open(os.path.join(src, "annotations.csv"), mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            annotations = {row[0]: row[1] for row in reader}

        for sample in os.listdir(src):
            if sample not in annotations:
                if not os.path.isdir(os.path.join(src, sample)):
                    continue
                print(f'WARNING: Sample {sample} missing label. Update annotations.csv before pushing.')
                return
        print(f'INFO: No missing labels.')
            
        for sample in os.listdir(src):
            dir = os.path.join(src, sample)
            if os.path.isdir(dir):
                self.upload_sample(dir, annotations[sample])
                self.move_sample(sample)
        print('PUSHING: Pushed')

        self.update_annotations('train')
        self.update_annotations('test')

        os.remove(os.path.join(self.tmp_path, "annotations.csv"))


    def check_integrity(self):
        test_blobs = [int(blob.name.split('/')[1]) for blob in self.container_client.walk_blobs(name_starts_with="test/", delimiter='/') if blob.name.endswith('/')]
        train_blobs = [int(blob.name.split('/')[1]) for blob in self.container_client.walk_blobs(name_starts_with="train/", delimiter='/') if blob.name.endswith('/')]

        next = 0
        with open(os.path.join(self.dataset_path, "annotations_test.csv"), mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            annotations = {row[0]: row[1] for row in reader}

        for sample in sorted(test_blobs):
            if sample != next:
                print(f'WARNING: Missing test sample {sample}. Replacing with blank.')
                with open('data_collection/blank.jpg', "rb") as data:
                    blob_client = self.container_client.get_blob_client(f'test/{sample}/0.jpg')
                    blob_client.upload_blob(data)
                    with open(f'{self.dataset_path}/annotations_test.csv', "a") as f:
                        f.write(f'{sample},blank\n')

            elif str(sample) not in annotations:
                print(f'WARNING: Missing label for test sample {sample}. Replacing with blank.')
                with open(f'{self.dataset_path}/annotations_test.csv', "a") as f:
                    f.write(f'{sample},blank\n')

            next += 5

        next = 1
        with open(os.path.join(self.dataset_path, "annotations_train.csv"), mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            annotations = {row[0]: row[1] for row in reader}

        for sample in sorted(train_blobs):
            if sample != next:
                print(f'WARNING: Missing train sample {sample}. Replacing with blank.')
                with open('data_collection/blank.jpg', "rb") as data:
                    blob_client = self.container_client.get_blob_client(f'train/{sample}/0.jpg')
                    blob_client.upload_blob(data)
                    with open(f'{self.dataset_path}/annotations_train.csv', "a") as f:
                        f.write(f'{sample},blank\n')

            elif str(sample) not in annotations:
                print(f'WARNING: Missing label for train sample {sample}. Replacing with blank.')
                with open(f'{self.dataset_path}/annotations_train.csv', "a") as f:
                    f.write(f'{sample},blank\n')

            next += 1
            if next % 5 == 0:
                next += 1

        self.update_annotations('train')
        self.update_annotations('test')
        print('INFO: Data integrity verified')