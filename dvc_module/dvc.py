import argparse

from data_collection import DataCollector
from data_upload import DataUploader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default=None, help='Command to execute (rec, push, pull)')
    parser.add_argument('--dataset_path', type=str, default='data/RGB', help='Pull destination path')
    parser.add_argument('--tmp_path', type=str, default='dvc_module/tmp', help='Push source path')

    return parser.parse_args()


def main():
    args = get_args()

    command = args.command
    dataset_path = args.dataset_path
    tmp_path = args.tmp_path

    collector = DataCollector(tmp_path=tmp_path)
    uploader = DataUploader(dataset_path=dataset_path, tmp_path=tmp_path)

    match command:
        case 'rec':
            collector.record_samples()

        case 'push':
            uploader.push(src=tmp_path)

        case 'pull':
            uploader.pull('train')
            uploader.pull('test')
            uploader.check_integrity()


if __name__ == '__main__':
    main()