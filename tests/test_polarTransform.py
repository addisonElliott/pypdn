import unittest
import numpy as np
import polarTransform
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util import loadImage, assert_image_equal


class TestPolarConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')
        self.horizontalLinesImage = loadImage('horizontalLines.png')
        self.checkerboardImage = loadImage('checkerboard.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

    def test_default(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (802, 1600))

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_defaultCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([400, 304]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 503)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.shortAxisApexImage.shape)
        self.assertEqual(ptSettings.polarImageSize, (800, 1600))

        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage_centerMiddle)

    def test_notNumpyArrayCenter(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=[401, 365])
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=(401, 365))
        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        np.testing.assert_almost_equal(polarImage, self.shortAxisApexPolarImage)

    def test_RGBA(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (256, 1024))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage)

    def test_IFRadius(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (99, 1024))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled2)

    def test_IFRadiusAngle(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (99, 384))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled3)

    def test_IFRadiusAngleScaled(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, initialRadius=30,
                                                                    finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                    finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                    angleSize=700)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (140, 700))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

    def test_settings(self):
        polarImage1, ptSettings1 = polarTransform.convertToPolarImage(self.verticalLinesImage,
                                                                      initialRadius=30,
                                                                      finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                      finalAngle=5 / 4 * np.pi, radiusSize=140,
                                                                      angleSize=700)

        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, settings=ptSettings1)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, self.verticalLinesImage.shape[0:2])
        self.assertEqual(ptSettings.polarImageSize, (140, 700))

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImage_scaled)

        polarImage2 = ptSettings1.convertToPolarImage(self.verticalLinesImage)
        np.testing.assert_almost_equal(polarImage2, self.verticalLinesPolarImage_scaled)


class TestCartesianConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')
        self.horizontalLinesImage = loadImage('horizontalLines.png')
        self.checkerboardImage = loadImage('checkerboard.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.verticalLinesCartesianImage = loadImage('verticalLinesCartesianImage.png')
        self.verticalLinesCartesianImage_scaled = loadImage('verticalLinesCartesianImage_scaled.png')
        self.verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')
        self.verticalLinesCartesianImage_scaled3 = loadImage('verticalLinesCartesianImage_scaled3.png')

        self.shortAxisApexCartesianImage = loadImage('shortAxisApexCartesianImage.png')
        self.shortAxisApexCartesianImage2 = loadImage('shortAxisApexCartesianImage2.png')

    def test_defaultCenter(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.shortAxisApexPolarImage,
                                                                            center=[401, 365], imageSize=[608, 800],
                                                                            finalRadius=543)

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.shortAxisApexCartesianImage)

    def test_notNumpyArrayCenter(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.shortAxisApexPolarImage_centerMiddle,
                                                                            imageSize=[608, 800], finalRadius=503)

        np.testing.assert_array_equal(ptSettings.center, np.array([400, 304]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 503)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage_centerMiddle.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.shortAxisApexCartesianImage2)

    def test_RGBA(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage,
                                                                            center=[128, 128],
                                                                            imageSize=[256, 256], finalRadius=182)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 182)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (256, 256))
        self.assertEqual(ptSettings.polarImageSize, self.verticalLinesPolarImage.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImage)

    def test_IFRadius(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled2,
                                                                            center=[128, 128], imageSize=[256, 256],
                                                                            initialRadius=30,
                                                                            finalRadius=100)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (256, 256))
        self.assertEqual(ptSettings.polarImageSize, self.verticalLinesPolarImage_scaled2.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImage_scaled2)

    def test_IFRadiusAngle(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled3,
                                                                            initialRadius=30,
                                                                            finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                            finalAngle=5 / 4 * np.pi, center=[128, 128],
                                                                            imageSize=[256, 256])

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (256, 256))
        self.assertEqual(ptSettings.polarImageSize, self.verticalLinesPolarImage_scaled3.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImage_scaled3)

    def test_IFRadiusAngleScaled(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled,
                                                                            initialRadius=30, finalRadius=100,
                                                                            initialAngle=2 / 4 * np.pi,
                                                                            finalAngle=5 / 4 * np.pi,
                                                                            imageSize=[256, 256],
                                                                            center=[128, 128])

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (256, 256))
        self.assertEqual(ptSettings.polarImageSize, self.verticalLinesPolarImage_scaled.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImage_scaled)

    def test_settings(self):
        cartesianImage1, ptSettings1 = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled,
                                                                              initialRadius=30, finalRadius=100,
                                                                              initialAngle=2 / 4 * np.pi,
                                                                              finalAngle=5 / 4 * np.pi,
                                                                              imageSize=[256, 256],
                                                                              center=[128, 128])

        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled,
                                                                            settings=ptSettings1)

        np.testing.assert_array_equal(ptSettings.center, np.array([128, 128]))
        self.assertEqual(ptSettings.initialRadius, 30)
        self.assertEqual(ptSettings.finalRadius, 100)
        self.assertEqual(ptSettings.initialAngle, 2 / 4 * np.pi)
        self.assertEqual(ptSettings.finalAngle, 5 / 4 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (256, 256))
        self.assertEqual(ptSettings.polarImageSize, self.verticalLinesPolarImage_scaled.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImage_scaled)

    def test_centerOrientationsWithImageSize(self):
        orientations = [
            ('bottom-left', np.array([0, 0]), [0, 128], [0, 128], [128, 256], [128, 256]),
            ('bottom-middle', np.array([128, 0]), [0, 128], [0, 256], [128, 256], [0, 256]),
            ('bottom-right', np.array([256, 0]), [0, 128], [128, 256], [128, 256], [0, 128]),

            ('middle-left', np.array([0, 128]), [0, 256], [0, 128], [0, 256], [128, 256]),
            ('middle-middle', np.array([128, 128]), [0, 256], [0, 256], [0, 256], [0, 256]),
            ('middle-right', np.array([256, 128]), [0, 256], [128, 256], [0, 256], [0, 128]),

            ('top-left', np.array([0, 256]), [128, 256], [0, 128], [0, 128], [128, 256]),
            ('top-middle', np.array([128, 256]), [128, 256], [0, 256], [0, 128], [0, 256]),
            ('top-right', np.array([256, 256]), [128, 256], [128, 256], [0, 128], [0, 128])
        ]

        for row in orientations:
            cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled2,
                                                                                center=row[0],
                                                                                imageSize=[256, 256],
                                                                                initialRadius=30,
                                                                                finalRadius=100)

            np.testing.assert_array_equal(ptSettings.center, row[1])
            self.assertEqual(ptSettings.cartesianImageSize, (256, 256))

            np.testing.assert_almost_equal(cartesianImage[row[2][0]:row[2][1], row[3][0]:row[3][1], :],
                                           self.verticalLinesCartesianImage_scaled2[row[4][0]:row[4][1],
                                           row[5][0]:row[5][1], :])

    def test_centerOrientationsWithoutImageSize(self):
        orientations = [
            ('bottom-left', (100, 100), np.array([0, 0]), [128, 228], [128, 228]),
            ('bottom-middle', (100, 200), np.array([100, 0]), [128, 228], [28, 228]),
            ('bottom-right', (100, 100), np.array([100, 0]), [128, 228], [28, 128]),

            ('middle-left', (200, 100), np.array([0, 100]), [28, 228], [128, 228]),
            ('middle-middle', (200, 200), np.array([100, 100]), [28, 228], [28, 228]),
            ('middle-right', (200, 100), np.array([100, 100]), [28, 228], [28, 128]),

            ('top-left', (100, 100), np.array([0, 100]), [28, 128], [128, 228]),
            ('top-middle', (100, 200), np.array([100, 100]), [28, 128], [28, 228]),
            ('top-right', (100, 100), np.array([100, 100]), [28, 128], [28, 128])
        ]

        for row in orientations:
            cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.verticalLinesPolarImage_scaled2,
                                                                                center=row[0],
                                                                                initialRadius=30,
                                                                                finalRadius=100)

            self.assertEqual(ptSettings.cartesianImageSize, row[1])
            np.testing.assert_array_equal(ptSettings.center, row[2])

            np.testing.assert_almost_equal(cartesianImage,
                                           self.verticalLinesCartesianImage_scaled2[row[3][0]:row[3][1],
                                           row[4][0]:row[4][1], :])


class TestPolarAndCartesianConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')
        self.horizontalLinesImage = loadImage('horizontalLines.png')
        self.checkerboardImage = loadImage('checkerboard.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.verticalLinesCartesianImage = loadImage('verticalLinesCartesianImage.png')
        self.verticalLinesCartesianImage_scaled = loadImage('verticalLinesCartesianImage_scaled.png')
        self.verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')
        self.verticalLinesCartesianImage_scaled3 = loadImage('verticalLinesCartesianImage_scaled3.png')

        self.shortAxisApexCartesianImage = loadImage('shortAxisApexCartesianImage.png')
        self.shortAxisApexCartesianImage2 = loadImage('shortAxisApexCartesianImage2.png')

        self.verticalLinesPolarImageBorders = loadImage('verticalLinesPolarImageBorders.png')
        self.verticalLinesCartesianImageBorders2 = loadImage('verticalLinesCartesianImageBorders2.png')
        self.verticalLinesPolarImageBorders3 = loadImage('verticalLinesPolarImageBorders3.png')
        self.verticalLinesCartesianImageBorders4 = loadImage('verticalLinesCartesianImageBorders4.png')

    def test_default(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage.shape[0:2])

        np.testing.assert_almost_equal(cartesianImage, self.shortAxisApexCartesianImage)

    def test_default2(self):
        polarImage1, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                     center=np.array([401, 365]), radiusSize=2000,
                                                                     angleSize=4000)

        cartesianImage = ptSettings.convertToCartesianImage(polarImage1)
        ptSettings.polarImageSize = self.shortAxisApexPolarImage.shape[0:2]
        polarImage = ptSettings.convertToPolarImage(cartesianImage)

        np.testing.assert_array_equal(ptSettings.center, np.array([401, 365]))
        self.assertEqual(ptSettings.initialRadius, 0)
        self.assertEqual(ptSettings.finalRadius, 543)
        self.assertEqual(ptSettings.initialAngle, 0.0)
        self.assertEqual(ptSettings.finalAngle, 2 * np.pi)
        self.assertEqual(ptSettings.cartesianImageSize, (608, 800))
        self.assertEqual(ptSettings.polarImageSize, self.shortAxisApexPolarImage.shape[0:2])

        assert_image_equal(polarImage, self.shortAxisApexPolarImage, 10)

    def test_borders(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, border='constant',
                                                                    borderVal=128.0)

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImageBorders)

        ptSettings.cartesianImageSize = (500, 500)
        ptSettings.center = np.array([250, 250])
        cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='constant', borderVal=255.0)

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImageBorders2)

    def test_borders2(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.verticalLinesImage, border='nearest')

        np.testing.assert_almost_equal(polarImage, self.verticalLinesPolarImageBorders3)

        ptSettings.cartesianImageSize = (500, 500)
        ptSettings.center = np.array([250, 250])
        cartesianImage = ptSettings.convertToCartesianImage(polarImage, border='nearest')

        np.testing.assert_almost_equal(cartesianImage, self.verticalLinesCartesianImageBorders4)


class TestPointConversion(unittest.TestCase):
    def setUp(self):
        self.shortAxisApexImage = loadImage('shortAxisApex.png')
        self.verticalLinesImage = loadImage('verticalLines.png')
        self.horizontalLinesImage = loadImage('horizontalLines.png')
        self.checkerboardImage = loadImage('checkerboard.png')

        self.shortAxisApexPolarImage = loadImage('shortAxisApexPolarImage.png')
        self.shortAxisApexPolarImage_centerMiddle = loadImage('shortAxisApexPolarImage_centerMiddle.png')
        self.verticalLinesPolarImage = loadImage('verticalLinesPolarImage.png')
        self.verticalLinesPolarImage_scaled = loadImage('verticalLinesPolarImage_scaled.png')
        self.verticalLinesPolarImage_scaled2 = loadImage('verticalLinesPolarImage_scaled2.png')
        self.verticalLinesPolarImage_scaled3 = loadImage('verticalLinesPolarImage_scaled3.png')

        self.verticalLinesCartesianImage = loadImage('verticalLinesCartesianImage.png')
        self.verticalLinesCartesianImage_scaled = loadImage('verticalLinesCartesianImage_scaled.png')
        self.verticalLinesCartesianImage_scaled2 = loadImage('verticalLinesCartesianImage_scaled2.png')
        self.verticalLinesCartesianImage_scaled3 = loadImage('verticalLinesCartesianImage_scaled3.png')

        self.shortAxisApexCartesianImage = loadImage('shortAxisApexCartesianImage.png')
        self.shortAxisApexCartesianImage2 = loadImage('shortAxisApexCartesianImage2.png')

        self.verticalLinesPolarImageBorders = loadImage('verticalLinesPolarImageBorders.png')
        self.verticalLinesCartesianImageBorders2 = loadImage('verticalLinesCartesianImageBorders2.png')
        self.verticalLinesPolarImageBorders3 = loadImage('verticalLinesPolarImageBorders3.png')
        self.verticalLinesCartesianImageBorders4 = loadImage('verticalLinesCartesianImageBorders4.png')

    def test_polarConversion(self):
        polarImage, ptSettings = polarTransform.convertToPolarImage(self.shortAxisApexImage,
                                                                    center=np.array([401, 365]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([401, 365]), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([[401, 365], [401, 365]]),
                                      np.array([[0, 0], [0, 0]]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage((401, 365)), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(((401, 365), (401, 365))),
                                      np.array([[0, 0], [0, 0]]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(np.array([401, 365])), np.array([0, 0]))
        np.testing.assert_array_equal(ptSettings.getPolarPointsImage(np.array([[401, 365], [401, 365]])),
                                      np.array([[0, 0], [0, 0]]))

        np.testing.assert_array_equal(ptSettings.getPolarPointsImage([[451, 365], [401, 400], [348, 365], [401, 305]]),
                                      np.array([[50 * 802 / 543, 0], [35 * 802 / 543, 400], [53 * 802 / 543, 800],
                                                [60 * 802 / 543, 1200]]))

    def test_cartesianConversion(self):
        cartesianImage, ptSettings = polarTransform.convertToCartesianImage(self.shortAxisApexPolarImage,
                                                                            center=[401, 365], imageSize=[608, 800],
                                                                            finalRadius=543)

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage([0, 0]), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage([[0, 0], [0, 0]]),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage((0, 0)), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(((0, 0), (0, 0))),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(np.array([0, 0])), np.array([401, 365]))
        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(np.array([[0, 0], [0, 0]])),
                                      np.array([[401, 365], [401, 365]]))

        np.testing.assert_array_equal(ptSettings.getCartesianPointsImage(
            np.array([[50 * 802 / 543, 0], [35 * 802 / 543, 400], [53 * 802 / 543, 800],
                      [60 * 802 / 543, 1200]])), np.array([[451, 365], [401, 400], [348, 365], [401, 305]]))

    if __name__ == '__main__':
        unittest.main()
