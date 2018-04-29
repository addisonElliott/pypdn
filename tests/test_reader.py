import os
import sys
import unittest
from pdn.reader import *
import imageio
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# TODO Write reader unit tests

class TestReader(unittest.TestCase):
    def setUp(self):
        self.layerVisibleTest1 = imageio.imread('./data/layerVisibleTest1.png', ignoregamma=True)
        self.layerVisibleTest2 = imageio.imread('./data/layerVisibleTest2.png', ignoregamma=True)
        self.layerVisibleTest3 = imageio.imread('./data/layerVisibleTest3.png', ignoregamma=True)
        self.flattenNormalTest = imageio.imread('./data/flattenNormalTest.png', ignoregamma=True)
        self.flattenNormalTest2 = imageio.imread('./data/flattenNormalTest2.png', ignoregamma=True)

    def test_read(self):
        layeredImage = read('./data/Untitled3.pdn')

        self.assertEqual(layeredImage.width, 800)
        self.assertEqual(layeredImage.height, 600)

        self.assertEqual(layeredImage.version.__class__.__name__, 'System_Version')
        self.assertTrue(hasattr(layeredImage.version, 'Major'))
        self.assertTrue(hasattr(layeredImage.version, 'Minor'))
        self.assertTrue(hasattr(layeredImage.version, 'Build'))
        self.assertTrue(hasattr(layeredImage.version, 'Revision'))
        self.assertIsInstance(layeredImage.version.Major, int)
        self.assertIsInstance(layeredImage.version.Minor, int)
        self.assertIsInstance(layeredImage.version.Build, int)
        self.assertIsInstance(layeredImage.version.Revision, int)

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        self.assertEqual(layer.name, 'Background')
        self.assertEqual(layer.visible, True)
        self.assertEqual(layer.isBackground, True)
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)
        np.testing.assert_equal(layer.image, self.layerVisibleTest1)

        layer = layeredImage.layers[1]
        self.assertEqual(layer.name, 'Layer 2')
        self.assertEqual(layer.visible, True)
        self.assertEqual(layer.isBackground, False)
        self.assertEqual(layer.opacity, 161)
        self.assertEqual(layer.blendMode, BlendType.Additive)
        # Ignore the alpha component because it should be 255 and not 161 like in image that is flattened
        np.testing.assert_equal(layer.image[:, :, 0:3], self.layerVisibleTest2[:, :, 0:3])

    def test_flatten_normal1(self):
        layeredImage = read('./data/Untitled2.pdn')

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        layer = layeredImage.layers[1]
        layer.visible = False
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        image = layeredImage.flatten(asByte=True)
        np.testing.assert_equal(image, self.flattenNormalTest2)

    def test_flatten_normal2(self):
        layeredImage = read('./data/Untitled2.pdn')

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        layer = layeredImage.layers[1]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        image = layeredImage.flatten(asByte=True)
        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenNormalTest, rtol=2)

    def test_flatten_notasbyte(self):
        layeredImage = read('./data/Untitled2.pdn')

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        layer = layeredImage.layers[1]
        layer.visible = False
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        image = layeredImage.flatten(asByte=False)
        np.testing.assert_equal(image, self.flattenNormalTest2 / 255.)


if __name__ == '__main__':
    unittest.main()
