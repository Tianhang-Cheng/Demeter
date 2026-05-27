import argparse
import glob
import os

import imageio.v3 as iio
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Rename leaf images to zero-padded indices.')
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Folder containing .jpg leaf images.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder = os.path.abspath(args.data_dir)

    files = glob.glob(os.path.join(folder, '*.jpg'))
    files.sort()
    print(f'Total files: {len(files)}')

    for i, file in enumerate(tqdm.tqdm(files)):
        if os.path.basename(file).split('.')[0].isdigit():
            print(f'{file} is a already renamed, skipping')
            continue

        iio.imread(file)
        os.rename(file, os.path.join(folder, f'{i:05d}.jpg'))
        print(f'{file} -> {i:05d}.jpg')


if __name__ == '__main__':
    main()
