import click
import pandas as pd
import util
from sklearn.model_selection import train_test_split
import os
import shutil


@click.command()
@click.option('--data_dir', help='Dir with images.')
@click.option('--labels_file', help='Labels file for images.')
@click.option('--train_size', help='Size of sampled training set.')
@click.option('--val_size', help='Size of sampled validation set.')
def main(data_dir, labels_file, train_size, val_size):
    util.check_required_program_args([data_dir, labels_file, train_size, val_size])
    train_size = float(train_size)
    val_size = float(val_size)
    df = pd.read_csv(labels_file)
    data_size = len(df)
    trainval_fraction = (train_size + val_size) / data_size
    X_trainval, _, y_trainval, _ = train_test_split(df.image.values, df.level.values, test_size=1 - trainval_fraction,
                                                    stratify=df.level.values)

    print('Original dataset frequencies')
    print(df.level.value_counts(normalize=True))
    val_fraction = val_size / (train_size + val_size)
    try:
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_fraction,
                                                          stratify=y_trainval)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_fraction,
                                                          stratify=None)

    train_df = pd.DataFrame(dict(image=X_train, level=y_train))
    val_df = pd.DataFrame(dict(image=X_val, level=y_val))

    dest_dir = os.path.dirname(os.path.abspath(data_dir)) + '/sample'
    im_size = int(os.path.basename(os.path.abspath(data_dir)).split('_')[1])

    dirs = [dest_dir, '%s/training_%d' % (dest_dir, im_size), '%s/validation_%d' % (dest_dir, im_size)]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print('Sampled training set frequencies')
    print(train_df.level.value_counts(normalize=True))

    print('Sampled validation set frequencies')
    print(train_df.level.value_counts(normalize=True))

    train_df.to_csv('%s/training_labels.csv' % dest_dir, index=False, header=True)
    val_df.to_csv('%s/validation_labels.csv' % dest_dir, index=False, header=True)

    for f in X_train:
        os.symlink("%s/%s.tiff" % (data_dir, f), "%s/%s.tiff" % (dirs[1], f))

    for f in X_val:
        os.symlink("%s/%s.tiff" % (data_dir, f), "%s/%s.tiff" % (dirs[2], f))


if __name__ == '__main__':
    main()
