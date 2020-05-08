import gzip
import struct

import numpy as np
from aenum import IntEnum
import warnings

from pypdn.nrbf import NRBF


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

    def fromClassName(className):
        lookupTable = {
            'PaintDotNet_UserBlendOps_NormalBlendOp': BlendType.Normal,
            'PaintDotNet_UserBlendOps_MultiplyBlendOp': BlendType.Multiply,
            'PaintDotNet_UserBlendOps_AdditiveBlendOp': BlendType.Additive,
            'PaintDotNet_UserBlendOps_ColorBurnBlendOp': BlendType.ColorBurn,
            'PaintDotNet_UserBlendOps_ColorDodgeBlendOp': BlendType.ColorDodge,
            'PaintDotNet_UserBlendOps_ReflectBlendOp': BlendType.Reflect,
            'PaintDotNet_UserBlendOps_GlowBlendOp': BlendType.Glow,
            'PaintDotNet_UserBlendOps_OverlayBlendOp': BlendType.Overlay,
            'PaintDotNet_UserBlendOps_DifferenceBlendOp': BlendType.Difference,
            'PaintDotNet_UserBlendOps_NegationBlendOp': BlendType.Negation,
            'PaintDotNet_UserBlendOps_LightenBlendOp': BlendType.Lighten,
            'PaintDotNet_UserBlendOps_DarkenBlendOp': BlendType.Darken,
            'PaintDotNet_UserBlendOps_ScreenBlendOp': BlendType.Screen,
            'PaintDotNet_UserBlendOps_XorBlendOp': BlendType.XOR
        }

        return lookupTable.get(className, BlendType.Normal)


class LayeredImage:
    __slots__ = ['width', 'height', 'version', 'layers']

    def __init__(self, width, height, version):
        self.width = width
        self.height = height
        self.version = version
        self.layers = []

    def __repr__(self):
        return 'pypdn.LayeredImage(width={0}, height={1}, version={2!r}, layers={3!r})'.format(self.width, self.height,
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
                normalizedImage = imageIntToFloat(layer.image)

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
            # Catch warnings to prevent the possible precision lost warning
            # Personally, I think the user should know that converting to a uint8 will cause precision lost from float
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                image = imageFloatToInt(image)

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
        return 'pypdn.Layer(name={0}, visible={1}, isBackground={2}, opacity={3}, blendMode={4!r})' \
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
                bpp = stride * 8 // layeredImage.width

                # Convert 1D array to image array
                if bpp == 32:
                    image = np.frombuffer(data, np.uint8).reshape((layeredImage.height, layeredImage.width, 4))
                    image[:, :, 0:3] = np.flip(image[:, :, 0:3], axis=-1)
                elif bpp == 24:
                    image = np.frombuffer(data, np.uint8).reshape((layeredImage.height, layeredImage.width, 3))
                    image[:, :, 0:2] = np.flip(image[:, :, 0:2], axis=-1)
                else:
                    raise PDNReaderError('Invalid bpp %i' % bpp)

                # type(bitmapLayer.properties.blendOp).__name__
                # PaintDotNet_UserBlendOps_NormalBlendOp

                if hasattr(layerProps, 'blendMode'):
                    blendType = BlendType(layerProps.blendMode.value__)
                elif hasattr(bitmapLayer, 'properties') and hasattr(bitmapLayer.properties, 'blendOp'):
                    blendType = BlendType.fromClassName(type(bitmapLayer.properties.blendOp).__name__)
                else:
                    blendType = BlendType.Normal

                # Setup layer with information and append to the layers list
                layer = Layer(layerProps.name, layerProps.visible, layerProps.isBackground, layerProps.opacity,
                              blendType, image)
                layeredImage.layers.append(layer)

            return layeredImage
        except (AttributeError):
            raise PDNReaderError('Unable to read fields in NRBF PDN file')


def blendingFunc(A, B, blendType):
    # See here for information on Paint.NET's blending functions:
    # https://github.com/rivy/OpenPDN/blob/master/src/Data/UserBlendOps.Generated.H.cs
    if blendType == BlendType.Normal:
        return B
    elif blendType == BlendType.Multiply:
        return A * B
    elif blendType == BlendType.Additive:
        return np.minimum(A + B, 1.0)
    elif blendType == BlendType.ColorBurn:
        with np.errstate(divide='ignore'):
            return np.where(B != 0.0, np.maximum(1.0 - ((1.0 - A) / B), 0.0), 0.0)
    elif blendType == BlendType.ColorDodge:
        with np.errstate(divide='ignore'):
            return np.where(B != 1.0, np.minimum(A / (1.0 - B), 1.0), 1.0)
    elif blendType == BlendType.Reflect:
        with np.errstate(divide='ignore'):
            return np.where(B != 1.0, np.minimum((A ** 2) / (1.0 - B), 1.0), 1.0)
    elif blendType == BlendType.Glow:
        with np.errstate(divide='ignore'):
            return np.where(A != 1.0, np.minimum((B ** 2) / (1.0 - A), 1.0), 1.0)
    elif blendType == BlendType.Overlay:
        return np.where(A < 0.5, 2 * A * B, 1.0 - (2 * (1.0 - A) * (1.0 - B)))
    elif blendType == BlendType.Difference:
        return np.abs(A - B)
    elif blendType == BlendType.Negation:
        return 1.0 - np.abs(1.0 - A - B)
    elif blendType == BlendType.Lighten:
        return np.maximum(A, B)
    elif blendType == BlendType.Darken:
        return np.minimum(A, B)
    elif blendType == BlendType.Screen:
        return A + B - A * B
    elif blendType == BlendType.XOR:
        # XOR is meant for integer numbers, so must convert to uint8 first
        # Catch warnings to prevent the precision lost warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return imageIntToFloat(imageFloatToInt(A) ^ imageFloatToInt(B))


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

def imageIntToFloat(image):
    """Convert a numpy array representing an image to an array of floats

    Args:
        image (np.array(int)): A numpy array of int values
    """
    return image/255


def imageFloatToInt(image):
    """Convert a numpy array representing an image to an array of ints

    Args:
        image (np.array(float)): A numpy array of float values
    """
    return (image*255).astype(np.uint8)
