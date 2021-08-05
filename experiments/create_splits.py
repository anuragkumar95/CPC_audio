import glob
import os
import pathlib
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create train/validation scripts given a directory.')
    parser.add_argument('dir', type=pathlib.Path,
                        help='directory where to find the files')
    parser.add_argument('out_dir', type=pathlib.Path,
                        help='directory where to save the split files')

    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='ratio of files to put in the training subset, the rest will be reserved for validation')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(0)

    all = glob.glob(f'{args.dir}/**/*.flac')
    all = [pathlib.Path(f).stem for f in all]
    random.shuffle(all)
    out_train = f'{args.out_dir}/train_split.txt'
    out_valid = f'{args.out_dir}/test_split.txt'

    train_len = int(args.train_ratio * len(all))
    print(f'''Writing to:
    fnm train {out_train}
    fnm valid {out_valid}
    # train {train_len}
    # valid {len(all)-train_len}''')

    with open(out_train, 'w') as f:
        f.writelines([f'{l}\n' for l in all[:train_len]])

    with open(out_valid, 'w') as f:
        f.writelines([f'{l}\n' for l in all[train_len:]])
