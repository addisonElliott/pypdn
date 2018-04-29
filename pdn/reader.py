# TODO This is the goal API we are going for
# import pdn
#
# pdn.read(xxx)
#
#
# pdn.write(xxx)

import gzip
import struct

import numpy as np
from aenum import IntEnum

from pdn.nrbf import NRBF


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

    def flattenImage(self, useAll=False, applyBlendMode=True):
        # Take each layer and apply to get the resulting image
        pass


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


filename = '../tests/data/Untitled2.pdn'
print(read(filename))
