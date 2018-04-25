# TODO This is the goal API we are going for
# import pdn
#
# pdn.read(xxx)
#
#
# pdn.write(xxx)

import struct
import nrbf
import numpy as np
import gzip
import matplotlib.pyplot as plt

class PDNReaderError(Exception):
    """Exceptions for Paint.NET Reader module."""
    pass

def _readStr(fh, numBytes, encoding='ascii'):
    rawData = fh.read(numBytes)
    if len(rawData) != numBytes:
        raise PDNReaderError('Cannot read %d bytes to decode string. File may be corrupted.' % numBytes)

    return rawData.decode(encoding)


# Open filename
# Read magic str
# Read header length
# Read header

with open(filename, 'rb') as fh:
    # Read header section!
    magicStr = _readStr(fh, 4)
    print(magicStr)
    if magicStr != 'PDN3':
        raise PDNReaderError('Invalid magic string for PDN file: %s' % magicStr)

    headerSizeStr = fh.read(3) + b'\x00'
    if len(headerSizeStr) != 4:
        raise PDNReaderError('Unable to read header size. File may be corrupted.')

    headerSize = struct.unpack('<i', headerSizeStr)[0]
    print(headerSize)

    header = _readStr(fh, headerSize, 'utf-8')
    print(header)

    # TODO Parse header eventually

    if fh.read(2) != b'\x00\x01':
        raise PDNReaderError('Invalid data indicator bytes. File may be corrupted')

    print('')
    print('')

    # Read data now
    #     data = deserializeDotNET(fh)
    serial = nrbf.serialization(fh)
    data = serial.read_stream()

    print(type(data))
    print(len(data))

    for key in data:
        print(type(key))
    print('')
    print('')

    print(type(serial))
    print(serial.__dict__.keys())

    print('')
    print('')

    # Width/height & version
    print(data.width)
    print(data.height)
    print(data.savedWith)
    print(data.__dict__.keys())
    print('')
    print('')

    print(type(data.layers))
    print(data.layers.__dict__.keys())
    print(len(data.layers))
    print(data.layers.ArrayList__size)
    print('')
    print('')

    layers = data.layers.ArrayList__items
    layer = layers[0]
    print(type(layer))
    print(layer.__dict__.keys())
    print(layer.Layer_properties)
    print('')
    print('')

    surface = layer.surface
    print(type(surface))
    print(surface.__dict__.keys())
    print(surface.width)
    print(surface.height)
    print(surface.stride)
    print('')
    print('')

    scan = surface.scan0
    print(type(scan))
    print(scan.__dict__.keys())
    print(scan.length64)
    print(scan.hasParent)
    print(scan.deferred)
    print('')
    print('')

    print(fh.tell())

    # Begin reading data now....
    # Look here: https://github.com/rivy/OpenPDN/blob/cca476b0df2a2f70996e6b9486ec45327631568c/src/Core/MemoryBlock.cs

    data = bytearray([0] * scan.length64)

    # Format version of the data, 0 = GZIP compressed, 1 = uncompressed
    formatVersion = struct.unpack('>B', fh.read(1))[0]
    # Size of each chunk in the destination (buffer where we store data)
    chunkSize = struct.unpack('>I', fh.read(4))[0]

    # Get total number of chunks which is total length divided by chunkSize
    # Keep track of the chunks found in case the file is corrupted
    chunkCount = np.ceil(scan.length64 / chunkSize).astype(np.uint32)
    chunksFound = [False] * chunkCount

    for x in range(chunkCount):
        # Read the chunk number, they are not necessarily in order
        chunkNumber = struct.unpack('>I', fh.read(4))[0]

        if chunkNumber >= chunkCount:
            raise PDNReaderError('Chunk number read from stream is out of bounds: %i %i. File may be corrupted'
                                 % (chunkNumber, chunkCount))

        if chunksFound[chunkNumber]:
            raise PDNReaderError('Chunk number %i was already encountered. File may be corrupted' % chunkNumber)

        # Read the size of the data from memory
        # This is not necessary the same size as chunkSize if the data is compressed
        dataSize = struct.unpack('>I', fh.read(4))[0]

        # Calculate the chunk offset
        chunkOffset = chunkNumber * chunkSize

        # Mark this chunk as found
        chunksFound[chunkNumber] = True

        # The chunk size is a maximum value and should be limited for the last chunk where it might not be exactly equal
        # to the chunk size
        currentChunkSize = np.min((chunkSize, scan.length64 - chunkOffset))

        # Read the chunk data
        rawData = fh.read(dataSize)

        if formatVersion == 0:
            decompressedData = gzip.decompress(rawData)
            data[chunkOffset:chunkOffset + chunkSize] = decompressedData
        else:
            data[chunkOffset:chunkOffset + dataSize] = rawData

        #         if (formatVersion == 0)
        #         {
        #             DecompressChunkParms parms = new DecompressChunkParms(compressedBytes, thisChunkSize, chunkOffset, context, exceptions);
        #             threadPool.QueueUserWorkItem(callback, parms);
        #         }
        #         else
        #         {
        #             fixed (byte *pbSrc = compressedBytes)
        #             {
        #                 Memory.Copy((void *)((byte *)this.VoidStar + chunkOffset), (void *)pbSrc, thisChunkSize);
        #             }
        #         }

        print(chunkNumber, dataSize)

    print(formatVersion, chunkSize)
    print(chunkCount, chunksFound)
    print('')
    print('')

    # Okay, so now take the data and convert it to a PNG image to display
    bpp = surface.stride * 8 / surface.width

    if bpp == 32:
        image = np.frombuffer(data, np.uint8).reshape((surface.height, surface.width, 4))
        #         image = np.flip(image, axis=2)
        image[:, :, 0:3] = np.flip(image[:, :, 0:3], axis=-1)
    elif bpp == 24:
        pass
    else:
        raise PDNReaderError('Invalid bpp %i' % bpp)

    plt.imshow(image)