import click
import matplotlib.pyplot as plt

from tefla.da import data
import numpy as np


@click.command()
@click.option('--images_dir', required=True, help='Directory with Test Images')
def visualize(images_dir):
    files = data.get_image_files(images_dir)
    original_images = data.load_images(files)
    num = len(original_images)
    tfs = list()
    tfs.append(data.build_augmentation_transform((1, 1), 0, 0, (0, 200), False))
    tfs.append(data.build_augmentation_transform((1, 1), 45, 0, (0, 0), False))
    tfs.append(data.build_augmentation_transform((1, 1), 45, 0, (0, 200), False))
    tfs.append(data.build_augmentation_transform((1, 1), 0, 45, (0, 0), False))
    num_cols = len(tfs) + 1

    for j, file in enumerate(files):
        plt.subplot(num, num_cols, j * num_cols + 1)
        imshow(plt, original_images[j])
        for i, tf in enumerate(tfs, start=2):
            plt.subplot(num, num_cols, j * num_cols + i)
            img = data.load_augment(file, data.image_no_preprocessing, 512, 512, False, transform=tf)
            imshow(plt, img)
    plt.show()


def imshow(plt, image):
    show_img = image.astype(np.uint8)
    plt.imshow(show_img)


def dump_transform(tf):
    print("Transform: translation: %s, rotation: %s, scale: %s, shear: %s" % (
        tf.translation, tf.rotation, tf.scale, tf.shear))


if __name__ == '__main__':
    visualize()
