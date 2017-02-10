
from PIL import Image, ImageFilter
import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import filters
from skimage import measure
from skimage.color import rgb2gray
from skimage.morphology import closing, square
from tefla.da import data
import os
import shutil
plt.get_backend()


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


def find_boundary_reverse(count,size):
    retval=count.size-1
    if count[count.size-1]==size or count[count.size-1]==0:
        for i in range(np.array(count).size-2, 0, -1):
            if(count[i] != count[i+1]):
                retval=i
                break
    return retval

def find_boundary(count,size):
    retval=0
    if count[0]==size or count[0]==0:
        for i in range(0,np.array(count).size-2, 1):
            if(count[i] != count[i+1]):
                retval=i
                break
    return retval

def subplots(img,row_count,col_count,crop_img):
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 3])
    plt.subplot(gs[0])
    imshow(plt, img)

    plt.subplot(gs[1])
    plt.plot(np.array(row_count))

    plt.subplot(gs[2])
    plt.plot(np.array(col_count))

    plt.subplot(gs[3])
    imshow(plt,crop_img)
    plt.show()

#crops an image from both dark and light background
#works best on a single color background
def crop_image(fname,target_size):
    print('Processing image: %s' % fname)

    #otsu thresholding
    img = Image.open(fname)
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    gray_image = cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #storing white pixel in  each row and column in two arrays
    #these arrays are later used to find boundaries for cropping image
    row_white_pixel_count=np.count_nonzero(threshold2,axis=1)
    col_white_pixel_count=np.count_nonzero(threshold2,axis=0)

    #find x,y,w,h for cropping image
    y=find_boundary(row_white_pixel_count,col_white_pixel_count.size)
    h=find_boundary_reverse(row_white_pixel_count,col_white_pixel_count.size)
    x=find_boundary(col_white_pixel_count,row_white_pixel_count.size)
    w=find_boundary_reverse(col_white_pixel_count,row_white_pixel_count.size)
    crop_array = ba[y:h, x:w]

    #resize the image
    crop_img=Image.fromarray(crop_array)
    resized = crop_img.resize([target_size, target_size])


    #uncomment below line to see histogram of both white pixel vs rows and white pixel vs columns
    #subplots(threshold2, row_white_pixel_count, col_white_pixel_count, crop_img)
    return resized


def draw_top_regions(properties, n0):
    n = min(n0, len(properties))
    print('Drawing %d regions' % n)
    colors = ['red', 'blue', 'green']
    for i in range(n):
        region = properties[i]
        minr, minc, maxr, maxc = region.bbox
        print('Region bbox: %s' % str(region.bbox))
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=colors[i % len(colors)], linewidth=1)
        plt.gca().add_patch(rect)
        return minr, minc, maxr, maxc



def imshow(plt, image):
    show_img = image.astype(np.uint8)
    plt.imshow(show_img, cmap='gray')


images_dir = '../experiments/images/'

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
    img = crop_image(file,512)
    plt.imshow(img)

plt.show()
