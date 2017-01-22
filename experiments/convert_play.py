from PIL import Image, ImageFilter

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from skimage import measure
from skimage.color import rgb2gray
from skimage.morphology import closing, square
from tefla.da import data


def convert(fname, target_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img, fname)

    cropped = img.crop(bbox)
    resized = cropped.resize([target_size, target_size])
    return np.array(resized)


def square_bbox(img, fname):
    print("square bbox conversion done for image: %s" % fname)
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_new(fname, target_size):
    print('Processing image: %s' % fname)
    img = Image.open(fname)
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    ba_gray = rgb2gray(ba)
    val = filters.threshold_otsu(ba_gray)
    # foreground = (ba_gray > val).astype(np.uint8)
    foreground = closing(ba_gray > val, square(3))
    # kernel = morphology.rectangle(5, 5)
    # foreground = morphology.binary_dilation(foreground, kernel)
    labels = measure.label(foreground)
    properties = measure.regionprops(labels)
    properties = sorted(properties, key=lambda p: p.area, reverse=True)
    # draw_top_regions(properties, 3)
    # return ba
    bbox = properties[0].bbox
    bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
    cropped = img.crop(bbox)
    resized = cropped.resize([target_size, target_size])
    return np.array(resized)


def convert_new_regions(fname, target_size):
    print('Processing image: %s' % fname)
    img = Image.open(fname)
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    ba_gray = rgb2gray(ba)
    val = filters.threshold_otsu(ba_gray)
    # foreground = (ba_gray > val).astype(np.uint8)
    foreground = closing(ba_gray > val, square(3))
    # kernel = morphology.rectangle(5, 5)
    # foreground = morphology.binary_dilation(foreground, kernel)
    labels = measure.label(foreground)
    properties = measure.regionprops(labels)
    properties = sorted(properties, key=lambda p: p.area, reverse=True)
    draw_top_regions(properties, 3)
    return ba


def draw_top_regions(properties, n0):
    n = min(n0, len(properties))
    print('Drawing %d regions' % n)
    colors = ['red', 'red', 'red']
    for i in range(n):
        region = properties[i]
        minr, minc, maxr, maxc = region.bbox
        print('Region bbox: %s' % str(region.bbox))
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=colors[i % len(colors)], linewidth=1)
        plt.gca().add_patch(rect)


def imshow(plt, image):
    show_img = image.astype(np.uint8)
    plt.imshow(show_img, cmap='gray')


images_dir = '/media/lalit/data/eyepacs-dr/make-testset/viz/images'

files = data.get_image_files(images_dir)
original_images = data.load_images(files)
num = len(original_images)
num_cols = 4

for j, file in enumerate(files):
    plt.subplot(num, num_cols, j * num_cols + 1)
    imshow(plt, original_images[j])

    plt.subplot(num, num_cols, j * num_cols + 2)
    img = convert(file, 512)
    imshow(plt, img)

    plt.subplot(num, num_cols, j * num_cols + 3)
    img = convert_new(file, 512)
    imshow(plt, img)

    plt.subplot(num, num_cols, j * num_cols + 4)
    img = convert_new_regions(file, 512)
    imshow(plt, img)

plt.show()
