import numpy as np
from torchvision import transforms


def upscale_masks(masks, resolution=768):
    upscaled_masks = list()
    for mask in masks:
        image = transforms.functional.to_pil_image(mask)
        image = image.resize((resolution, resolution))
        upscaled_masks.append(image)

    return upscaled_masks


def test_upscale_masks():
    size = 16

    input_image1 = np.zeros((size, size, 1), np.uint8)
    input_image1[1, 1, 0] = 255
    input_image1[1, 2, 0] = 255
    input_image1[2, 1, 0] = 255
    input_image1[2, 2, 0] = 255

    result = upscale_masks([input_image1], 32)
    result = np.array(result[0])

    expected_image = np.zeros((size * 2, size * 2), np.uint8)
    expected_image[2, 2] = 255
    expected_image[2, 3] = 255
    expected_image[2, 4] = 255
    expected_image[2, 5] = 255

    expected_image[3, 2] = 255
    expected_image[3, 3] = 255
    expected_image[3, 4] = 255
    expected_image[3, 5] = 255

    expected_image[4, 2] = 255
    expected_image[4, 3] = 255
    expected_image[4, 4] = 255
    expected_image[4, 5] = 255

    expected_image[5, 2] = 255
    expected_image[5, 3] = 255
    expected_image[5, 4] = 255
    expected_image[5, 5] = 255

    assert np.array_equal(result, expected_image)
