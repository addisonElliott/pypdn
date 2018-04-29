import gzip
import struct

import matplotlib.pyplot as plt
import numpy as np
import skimage
from aenum import IntEnum

from pdn.nrbf import NRBF


# TODO Add API for writing PDN file

# TODO Write docstrings basic

class PDNReaderError(Exception):
    """Exceptions for Paint.NET Reader module."""
    pass


class BlendType(IntEnum):
    Normal = 0
    Multiply = 1
    Additive = 2
    ColorBurn = 3
    ColorDodge = 4
    Reflect = 5
    Glow = 6
    Overlay = 7
    Difference = 8
    Negation = 9
    Lighten = 10
    Darken = 11
    Screen = 12
    XOR = 13


class LayeredImage:
    __slots__ = ['width', 'height', 'version', 'layers']

    def __init__(self, width, height, version):
        self.width = width
        self.height = height
        self.version = version
        self.layers = []

    def __repr__(self):
        return 'pdn.LayeredImage(width={0}, height={1}, version={2!r}, layers={3!r})'.format(self.width, self.height,
                                                                                             self.version, self.layers)

    def flatten(self, useAll=False, applyBlendMode=True, asByte=False):
        if len(self.layers) == 0:
            return None

        # Create empty RGBA image to hold the flattened image
        # Use float datatype because that is how the calculations are done to prevent scaling
        # Image can be scaled at the end
        image = np.zeros((self.height, self.width, 4), dtype=float)

        for layer in self.layers:
            if useAll or layer.visible:
                blendMode = BlendType.Normal if not applyBlendMode else layer.blendMode

                # Operations must be done on float image!
                normalizedImage = skimage.img_as_float(layer.image)

                # Apply the layer opacity to the image here
                # If the image does not have an alpha component, extend the image to contain one
                if normalizedImage.shape[2] == 3:
                    alpha = np.ones(normalizedImage.shape[:-1], dtype=float)
                    normalizedImage = np.dstack((normalizedImage, alpha))

                # Now multiply the layer opacity by the alpha component of image
                normalizedImage[:, :, 3] = normalizedImage[:, :, 3] * (layer.opacity / 255.)

                # Take current image and apply the normalized image from the layer to it with specified blend mode
                image = applyBlending(image, normalizedImage, blendMode)

        # Paint.NET stores everything as uint8's
        # We had to convert to float to flatten the image and now we can convert back to uint8 if desired
        if asByte:
            image = skimage.img_as_ubyte(image)

        return image


class Layer:
    __slots__ = ['name', 'visible', 'isBackground', 'opacity', 'blendMode', 'image']

    def __init__(self, name, visible, isBackground, opacity, blendMode, image):
        self.name = name
        self.visible = visible
        self.isBackground = isBackground
        self.opacity = opacity
        self.blendMode = blendMode
        self.image = image

    def __repr__(self):
        return 'pdn.Layer(name={0}, visible={1}, isBackground={2}, opacity={3}, blendMode={4!r})' \
            .format(self.name, self.visible, self.isBackground, self.opacity, self.blendMode)


def read(filename):
    with open(filename, 'rb') as fh:
        # Begin by reading magic str and checking it
        magicStr = fh.read(4).decode('ascii')

        if magicStr != 'PDN3':
            raise PDNReaderError('Invalid magic string for PDN file: %s' % magicStr)

        headerSizeStr = fh.read(3) + b'\x00'
        if len(headerSizeStr) != 4:
            raise PDNReaderError('Unable to read header size. File may be corrupted.')

        # Read header size and the header
        headerSize = struct.unpack('<i', headerSizeStr)[0]
        header = fh.read(headerSize).decode('utf-8')

        # Note: The header does not contain any relevant information that is not contained in the NRBF part itself
        # Only section is the thumbnail image but that is not relevant here I do not believe

        if fh.read(2) != b'\x00\x01':
            raise PDNReaderError('Invalid data indicator bytes. File may be corrupted')

        nrbfData = NRBF(stream=fh)
        pdnDocument = nrbfData.getRoot()

        try:
            layeredImage = LayeredImage(pdnDocument.width, pdnDocument.height, pdnDocument.savedWith)

            # Cannot loop through items array because sometimes it is padded with null objects, size is right though
            for index in range(pdnDocument.layers.ArrayList__size):
                bitmapLayer = pdnDocument.layers.ArrayList__items[index]
                layerProps = bitmapLayer.Layer_properties

                # Read information from layer that is used to read the image data
                assert bitmapLayer.Layer_width == layeredImage.width
                assert bitmapLayer.Layer_height == layeredImage.height
                stride = bitmapLayer.surface.stride
                length = bitmapLayer.surface.scan0.length64

                # Begin reading the image data from the layer
                # Look here for a reference of how it is written:
                # https://github.com/rivy/OpenPDN/blob/cca476b0df2a2f70996e6b9486ec45327631568c/src/Core/MemoryBlock.cs

                # Empty array of the length, will be filled in later
                data = bytearray([0] * length)

                # Format version of the data, 0 = GZIP compressed, 1 = uncompressed
                formatVersion = struct.unpack('>B', fh.read(1))[0]
                # Size of each chunk in the destination (buffer where we store data)
                chunkSize = struct.unpack('>I', fh.read(4))[0]

                # Get total number of chunks which is total length divided by chunkSize
                # Keep track of the chunks found in case the file is corrupted
                chunkCount = np.ceil(length / chunkSize).astype(np.uint32)
                chunksFound = [False] * chunkCount

                for x in range(chunkCount):
                    # Read the chunk number, they are not necessarily in order
                    chunkNumber = struct.unpack('>I', fh.read(4))[0]

                    if chunkNumber >= chunkCount:
                        raise PDNReaderError(
                            'Chunk number read from stream is out of bounds: %i %i. File may be corrupted'
                            % (chunkNumber, chunkCount))

                    if chunksFound[chunkNumber]:
                        raise PDNReaderError(
                            'Chunk number %i was already encountered. File may be corrupted' % chunkNumber)

                    # Read the size of the data from memory
                    # This is not necessary the same size as chunkSize if the data is compressed
                    dataSize = struct.unpack('>I', fh.read(4))[0]

                    # Calculate the chunk offset
                    chunkOffset = chunkNumber * chunkSize

                    # Mark this chunk as found
                    chunksFound[chunkNumber] = True

                    # The chunk size is a maximum value and should be limited for the last chunk where it might not be
                    # exactly equal to the chunk size
                    actualChunkSize = np.min((chunkSize, length - chunkOffset))

                    # Read the chunk data
                    rawData = fh.read(dataSize)

                    if formatVersion == 0:
                        decompressedData = gzip.decompress(rawData)
                        data[chunkOffset:chunkOffset + actualChunkSize] = decompressedData
                    else:
                        data[chunkOffset:chunkOffset + actualChunkSize] = rawData

                # With the data read in as one large 1D array, we now format the data properly
                # Calculate the bits per pixel
                bpp = stride * 8 / layeredImage.width

                # Convert 1D array to image array
                if bpp == 32:
                    image = np.frombuffer(data, np.uint8).reshape((layeredImage.height, layeredImage.width, 4))
                    image[:, :, 0:3] = np.flip(image[:, :, 0:3], axis=-1)
                elif bpp == 24:
                    image = np.frombuffer(data, np.uint8).reshape((layeredImage.height, layeredImage.width, 3))
                    image[:, :, 0:2] = np.flip(image[:, :, 0:2], axis=-1)
                else:
                    raise PDNReaderError('Invalid bpp %i' % bpp)

                # Setup layer with information and append to the layers list
                layer = Layer(layerProps.name, layerProps.visible, layerProps.isBackground, layerProps.opacity,
                              BlendType(layerProps.blendMode.value__), image)
                layeredImage.layers.append(layer)

            return layeredImage
        except (AttributeError):
            raise PDNReaderError('Unable to read fields in NRBF PDN file')


def blendingFunc(A, B, blendType):
    # TODO Finish blending functions
    if blendType == BlendType.Normal:
        return B
    elif blendType == BlendType.Multiply:
        return A * B
    elif blendType == BlendType.Additive:
        return np.minimum(A + B, 1.0)
    elif blendType == BlendType.ColorBurn:
        pass
    elif blendType == BlendType.ColorDodge:
        pass
    elif blendType == BlendType.Reflect:
        pass
    elif blendType == BlendType.Glow:
        pass
    elif blendType == BlendType.Overlay:
        pass
    elif blendType == BlendType.Difference:
        pass
    elif blendType == BlendType.Negation:
        pass
    elif blendType == BlendType.Lighten:
        pass
    elif blendType == BlendType.Darken:
        pass
    elif blendType == BlendType.Screen:
        pass
    elif blendType == BlendType.XOR:
        pass


def applyBlending(A, B, blendType):
    # The generalized alpha composite with a changeable blending function is shown below
    #
    # G(A,a,B,b,F) = (a - ab)A + (b - ab)B + abF(A, B)
    #
    # Where:
    #   A = background color
    #   a = background alpha
    #   B = foreground color
    #   b = foreground alpha
    #   F is blending function (F(A,B))
    #
    # This is what Paint.NET uses to create the flattened image and I do so as well but with Numpy to make it
    # easier.
    aAlpha = A[:, :, 3]
    bAlpha = B[:, :, 3]
    abAlpha = aAlpha * bAlpha

    aColor = A[:, :, 0:3]
    bColor = B[:, :, 0:3]

    # Use generalized alpha compositing equation shown above to calculate the color components of the image
    # To be able to multiply along each component (R,G,B), we must add a new axis to the difference portion
    # That is what the None field indicates
    colorComponents = (aAlpha - abAlpha)[:, :, None] * aColor + \
                      (bAlpha - abAlpha)[:, :, None] * bColor + \
                      abAlpha[:, :, None] * blendingFunc(aColor, bColor, blendType)

    # #define COMPUTE_ALPHA(a, b, r) { INT_SCALE(a, 255 - (b), r); r += (b); }
    # alphaComponent = aAlpha * (1.0 - bAlpha) + bAlpha
    alphaComponent = aAlpha + bAlpha - abAlpha

    # Return RGBA image by combining RGB and A
    return np.dstack((colorComponents, alphaComponent))


# filename = '../tests/data/Untitled2.pdn'
# layeredImage = read(filename)
#
# image = layeredImage.flatten()
#
# plt.imshow(image)
# plt.show()
