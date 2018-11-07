import numpy as np


def rle_decode(rle_image, width, height):
    image_array = np.zeros(height*width)
    for rle_tuple in rle_image:
        number_of_pixels = int(rle_tuple[1]) - 1
        pixel = int(rle_tuple[0])
        image_array[pixel:pixel + number_of_pixels] = 1
    return np.reshape(image_array, (height, width), order='F')


def rle_encode():
    raise NotImplementedError
