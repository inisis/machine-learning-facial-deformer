import os
import argparse

# python generate_csv.py D:\data D:\data\csv

parser = argparse.ArgumentParser(description='Generate csv')
parser.add_argument(
    '--base_dir',
    default=r"D:\data\ray",
    type=str,
    help='Base dir to generate csv')
parser.add_argument(
    '--save_path',
    default=r"D:\data\ray\csv",
    type=str,
    help='Path to save generated csvs')


def find_all_files(base):
    for root, dirs, files in os.walk(base):
        for f in sorted(files, key=lambda x: int(x[:-4]) if x[:-4].isdigit() else -1):
            if f.endswith('.npy') and 'mesh' in root:
                fullname = os.path.join(root, f)
                yield fullname


def run(args):

    annotations = []

    for i in find_all_files(args.base_dir):
        annotation = i + ',' + i.replace('mesh', 'ctrl') + '\n'
        print(annotation)
        annotations.append(annotation)

    print(len(annotations))
    train_csv = os.path.join(args.save_path, 'train.csv')
    dev_csv = os.path.join(args.save_path, 'dev.csv')

    with open(train_csv, 'w') as file_train, open(dev_csv, 'w') as file_dev:
        for idx, annotation in enumerate(annotations):
            if idx % 50 == 0:
                file_dev.write(annotation)
            # else:
            file_train.write(annotation)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()