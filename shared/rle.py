import numpy as np


def rle_decode(rle_image, width, height):
    image_array = np.zeros(height * width)
    for (pixel, number_of_pixels) in rle_image:
        number_of_pixels = int(number_of_pixels)
        pixel = int(pixel)
        image_array[pixel:pixel + number_of_pixels] = 1
    return np.reshape(image_array, (height, width), order='F')


def rle_encode(image):
    assert len(image.shape) == 2
    image = np.rot90(image)
    image = np.flipud(image)
    flattened_image = image.flatten()
    length = len(flattened_image)
    rle_list = []
    encoding = False
    pixel_index = 0
    number_of_pixels = 0
    for index, pixel in enumerate(flattened_image):
        if encoding is True:
            if pixel > 0:
                number_of_pixels += 1
                if index == length - 1:
                    rle_list.append((pixel_index, number_of_pixels))
            else:
                encoding = False
                rle_list.append((pixel_index, number_of_pixels))
                pixel_index = 0
                number_of_pixels = 0
        if encoding is False:
            if pixel > 0:
                encoding = True
                pixel_index = index
                number_of_pixels = 1

    return rle_list


def test_rle_decode():
    rle_image = [(4, 2), (8, 2)]
    width = 32
    height = 32
    expected = np.zeros(height * width)
    expected[4] = 1
    expected[5] = 1
    expected[8] = 1
    expected[9] = 1
    expected = np.reshape(expected, (height, width), order='F')
    image = rle_decode(rle_image, width, height)
    assert np.array_equal(expected, image)


def test_rle_encode():
    expected = [(4, 2), (8, 2)]
    width = 32
    height = 32
    image = rle_decode(expected, width, height)
    rle_image = rle_encode(image)
    assert rle_image == expected
