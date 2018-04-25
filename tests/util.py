import os
import imageio
import numpy as np

dataDirectory = os.path.join(os.path.dirname(__file__), 'data')


def loadImage(filename, flipud=True, convertToGrayscale=False):
    image = imageio.imread(os.path.join(dataDirectory, filename), ignoregamma=True)

    if convertToGrayscale:
        image = image[:, :, 0]

    return np.flipud(image) if flipud else image


def saveImage(filename, image, flipud=True):
    imageio.imwrite(os.path.join(dataDirectory, filename), np.flipud(image) if flipud else image)


def assert_image_equal(desired, actual, diff):
    difference = np.abs(desired.astype(int) - actual.astype(int)).astype(np.uint8)

    assert (np.all(difference <= diff))
