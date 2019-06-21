import os
import sys
import unittest

import imageio

from pypdn.reader import *
from tests.util import getDataPath


class TestReader(unittest.TestCase):
    def setUp(self):
        self.layerVisibleTest1 = imageio.imread(getDataPath('layerVisibleTest1.png'), ignoregamma=True)
        self.layerVisibleTest2 = imageio.imread(getDataPath('layerVisibleTest2.png'), ignoregamma=True)
        self.layerVisibleTest3 = imageio.imread(getDataPath('layerVisibleTest3.png'), ignoregamma=True)
        self.flattenNormalTest = imageio.imread(getDataPath('flattenNormalTest.png'), ignoregamma=True)
        self.flattenNormalTest2 = imageio.imread(getDataPath('flattenNormalTest2.png'), ignoregamma=True)

        self.flattenMultiplyTest = imageio.imread(getDataPath('flattenMultiplyTest.png'), ignoregamma=True)
        self.flattenAdditiveTest = imageio.imread(getDataPath('flattenAdditiveTest.png'), ignoregamma=True)
        self.flattenColorBurnTest = imageio.imread(getDataPath('flattenColorBurnTest.png'), ignoregamma=True)
        self.flattenColorDodgeTest = imageio.imread(getDataPath('flattenColorDodgeTest.png'), ignoregamma=True)
        self.flattenReflectTest = imageio.imread(getDataPath('flattenReflectTest.png'), ignoregamma=True)
        self.flattenGlowTest = imageio.imread(getDataPath('flattenGlowTest.png'), ignoregamma=True)
        self.flattenOverlayTest = imageio.imread(getDataPath('flattenOverlayTest.png'), ignoregamma=True)
        self.flattenDifferenceTest = imageio.imread(getDataPath('flattenDifferenceTest.png'), ignoregamma=True)
        self.flattenNegationTest = imageio.imread(getDataPath('flattenNegationTest.png'), ignoregamma=True)
        self.flattenLightenTest = imageio.imread(getDataPath('flattenLightenTest.png'), ignoregamma=True)
        self.flattenDarkenTest = imageio.imread(getDataPath('flattenDarkenTest.png'), ignoregamma=True)
        self.flattenScreenTest = imageio.imread(getDataPath('flattenScreenTest.png'), ignoregamma=True)
        self.flattenXORTest = imageio.imread(getDataPath('flattenXORTest.png'), ignoregamma=True)

        self.flattenOpacityTest = imageio.imread(getDataPath('flattenOpacityTest.png'), ignoregamma=True)

        self.oldPDN3510Test = imageio.imread(getDataPath('oldPDN3510.png'), ignoregamma=True)

    def test_read(self):
        layeredImage = read(getDataPath('Untitled3.pdn'))

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
        layeredImage = read(getDataPath('Untitled2.pdn'))

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
        layeredImage = read(getDataPath('Untitled2.pdn'))

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
        np.testing.assert_allclose(image, self.flattenNormalTest, atol=2)

    def test_flatten_notasbyte(self):
        layeredImage = read(getDataPath('Untitled2.pdn'))

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
        np.testing.assert_allclose(image, self.flattenNormalTest2 / 255., atol=2)

    def test_flatten_multiply(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Multiply
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # plt.imshow(image)
        # plt.show()
        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenMultiplyTest, atol=2)

    def test_flatten_additive(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Additive
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenAdditiveTest, atol=2)

    def test_flatten_colorBurn(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.ColorBurn
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenColorBurnTest, atol=2)

    def test_flatten_colorDodge(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.ColorDodge
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenColorDodgeTest, atol=2)

    def test_flatten_reflect(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Reflect
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenReflectTest, atol=2)

    def test_flatten_glow(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Glow
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenGlowTest, atol=2)

    def test_flatten_overlay(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Overlay
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenOverlayTest, atol=2)

    def test_flatten_difference(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Difference
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenDifferenceTest, atol=2)

    def test_flatten_negation(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Negation
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenNegationTest, atol=2)

    def test_flatten_lighten(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Lighten
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenLightenTest, atol=2)

    def test_flatten_darken(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Darken
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenDarkenTest, atol=2)

    def test_flatten_screen(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.Screen
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenScreenTest, atol=2)

    def test_flatten_xor(self):
        layeredImage = read(getDataPath('FlattenBlendTest.pdn'))

        self.assertEqual(len(layeredImage.layers), 14)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        desiredLayer = BlendType.XOR
        for x in range(1, 14):
            layeredImage.layers[x].visible = (x == desiredLayer)

        self.assertEqual(layeredImage.layers[desiredLayer].opacity, 255)
        self.assertEqual(layeredImage.layers[desiredLayer].blendMode, desiredLayer)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenXORTest, atol=2)

    def test_flatten_opacity(self):
        layeredImage = read(getDataPath('Untitled3.pdn'))

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        layer.visible = True
        self.assertEqual(layer.opacity, 255)
        self.assertEqual(layer.blendMode, BlendType.Normal)

        layer = layeredImage.layers[1]
        layer.visible = True
        self.assertEqual(layer.opacity, 161)
        self.assertEqual(layer.blendMode, BlendType.Additive)

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image, self.flattenOpacityTest, atol=2)

    def test_old_pdn_format(self):
        layeredImage = read(getDataPath('oldPDN3510.pdn'))

        self.assertEqual(len(layeredImage.layers), 2)

        layer = layeredImage.layers[0]
        layer.visible = True

        layer = layeredImage.layers[1]
        layer.visible = True

        image = layeredImage.flatten(asByte=True)

        # May be rounding errors so do within 2 points
        np.testing.assert_allclose(image[:, :, 0:3], self.oldPDN3510Test[:, :, 0:3], atol=2)


if __name__ == '__main__':
    unittest.main()
