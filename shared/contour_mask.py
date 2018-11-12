import cv2
import numpy as np


def generate_ship_contour_masks(image):
    masks = list()
    width = image.shape[0]
    height = image.shape[1]
    _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        surface = np.zeros((width, height, 1), np.uint8)
        cv2.drawContours(surface, [contour], -1, 255, -1)
        masks.append(surface.reshape((width, height)))

    return masks


def test_generate_ship_contour_masks():
    size = 16

    expected_image1 = np.zeros((size, size), np.uint8)
    expected_image1[1, 1] = 255
    expected_image1[1, 2] = 255
    expected_image1[2, 1] = 255
    expected_image1[2, 2] = 255

    expected_image2 = np.roll(expected_image1, 4)

    image = expected_image1 + expected_image2

    [mask1, mask2] = generate_ship_contour_masks(image)

    assert np.array_equal(mask1, expected_image2)
    assert np.array_equal(mask2, expected_image1)
