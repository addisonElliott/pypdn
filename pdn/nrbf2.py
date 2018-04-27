import struct
import datetime
from decimal import Decimal

from pdn.util import *


# TODO Include note about inspiration for this
# Include link to the GitHub page

# TODO Do documentation and docstrings for this

class NRBFError(Exception):
    """Exceptions for NRBFError class."""
    pass


# Decorator which adds an enum value (an int or a length-one bytes object) and its associated reader function to a
# registration dict (either PrimitiveTypeReaders or RecordTypeReaders)
# The last two arguments, primitiveStructs and primitiveFormat are only specified for primitive types to create the
# Struct object. The Struct object parses the format and makes it quicker to parse the numbers by calling Struct.unpack
def _registerReader(typeDict, typeValue, primitiveStructs=None, primitiveFormat=None):
    assert isinstance(typeValue, int)

    def decorator(readFunction):
        if primitiveStructs is not None:
            primitiveStructs[typeValue] = struct.Struct(primitiveFormat)

        typeDict[typeValue] = readFunction
        return readFunction

    return decorator


class NRBF:
    # Dictionary that contains functions to call when reading records and primitives from NRBF file
    _RecordTypeReaders = {}
    _PrimitiveTypeReaders = {}

    # Dictionary that contains Struct objects for each of the primitive types. The Struct objects are precompiled to
    # speed up parsing the number
    _PrimitiveTypeStructs = {}

    # TODO Testing
    _PrimitiveTypeArrayReaders = {}

    def __init__(self, stream=None, filename=None):
        self.stream = None

        # If a stream or filename to be loaded is given, then call the read function
        # This makes the syntax cleaner to allow this creation of class and loading of data all in one line
        if stream is not None or filename is not None:
            self.read(stream, filename)

    def read(self, stream=None, filename=None):
        if stream is None and filename is not None:
            stream = open(filename, 'rb')

        # TODO Might want to assert seekable here!
        assert stream.readable()

        if self.stream is not None:
            # Means we are in the middle of reading or writing data, throw exception probably
            pass

        # TODO Check if this has already been loaded to reset stuff, maybe do by default anyway

        self.stream = stream

        # TODO Read header!
        # self._readHeader(stream)
        self._readRecord(stream)
        # TODO Loop through and read each item

        # Once we are done reading, we set the stream to None because it will not be used
        self.stream = None

    def write(self, stream=None, filename=None):
        if stream is None and filename is not None:
            stream = open(filename, 'wb')

        assert stream.writable()
        assert stream.seekable()
        pass

    def _readHeader(self, stream):
        pass

    def _readRecord(self, stream):
        recordType = self._readByte()

        return self._RecordTypeReaders[recordType](self)

    # region Primitive reader functions

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Boolean, _PrimitiveTypeStructs, '?')
    def _readBool(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Boolean].unpack(self.stream.read(1))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Byte, _PrimitiveTypeStructs, 'B')
    def _readByte(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Byte].unpack(self.stream.read(1))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Char)
    def _readChar(self):
        utf8Bytes = bytearray()
        while True:
            utf8Bytes += self.stream.read(1)
            try:
                return utf8Bytes.decode('utf-8')
            except UnicodeDecodeError:
                if len(utf8Bytes) > 4:
                    raise NRBFError('Invalid char read from NRBF file, longer than 4 bytes: {0}'.format(utf8Bytes))

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Decimal)
    def _read_Decimal(self):
        return Decimal(self._readString())

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Double, _PrimitiveTypeStructs, '<d')
    def _readDouble(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Double].unpack(self.stream.read(8))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Int16, _PrimitiveTypeStructs, '<h')
    def _readInt16(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Int16].unpack(self.stream.read(2))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Int32, _PrimitiveTypeStructs, '<i')
    def _readInt32(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Int32].unpack(self.stream.read(4))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Int64, _PrimitiveTypeStructs, '<q')
    def _readInt64(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Int64].unpack(self.stream.read(8))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.SByte, _PrimitiveTypeStructs, 'b')
    def _readSByte(self):
        return self._PrimitiveTypeStructs[PrimitiveType.SByte].unpack(self.stream.read(1))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Single, _PrimitiveTypeStructs, '<f')
    def _readSingle(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Single].unpack(self.stream.read(4))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.TimeSpan)
    def _readTimeSpan(self):
        # 64-bit integer that represents time span in increments of 100 nanoseconds
        # Divide by 10 to get into microseconds
        return datetime.timedelta(microseconds=self._readInt64() / 10)

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.DateTime)
    def _readDateTime(self):
        ticks = self._readUInt64()

        # Two MSB store kind, 0 = no timezone, 1 = UTC, 2 = local timezone
        kind = ticks >> 62
        # Remaining 62-bits are the number of 100ns increments from 12:00:00 January 1, 0001
        ticks &= (1 << 62) - 1
        # If negative, then reinterpret as 62-bit two's complement
        if ticks >= 1 << 61:
            ticks -= 1 << 62

        # Create a datetime that starts at the beginning and then increment it by the number of microseconds
        time = datetime.datetime(1, 1, 1)
        try:
            time += datetime.timedelta(microseconds=ticks / 10)
        except OverflowError:
            pass

        # Update datetime object to have the appropriate timezone
        # If kind is 1, then this is UTC and if kind is 2, then this is local timezone
        if kind == 1:
            time = time.replace(tzinfo=datetime.timezone.utc)
        elif kind == 2:
            LOCAL_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            time = time.replace(tzinfo=LOCAL_TIMEZONE)  # kind 2 is the local time zone

        return time

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.UInt16, _PrimitiveTypeStructs, '<H')
    def _readUInt16(self):
        return self._PrimitiveTypeStructs[PrimitiveType.UInt16].unpack(self.stream.read(2))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.UInt32, _PrimitiveTypeStructs, '<I')
    def _readUInt32(self):
        return self._PrimitiveTypeStructs[PrimitiveType.UInt32].unpack(self.stream.read(4))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.UInt64, _PrimitiveTypeStructs, '<Q')
    def _readUInt64(self):
        return self._PrimitiveTypeStructs[PrimitiveType.UInt64].unpack(self.stream.read(8))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Null)
    def _readNull(self):
        return None

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.String)
    def _readString(self):
        length = 0

        # Each bit range is 7 bits long with a maximum of 5 bytes
        for bit_range in range(0, 5 * 7, 7):
            byte = self._readByte()

            # Remove the last bit from the length (used to indicate if there is another byte to come)
            # Then shift the number to the appropiate bit range and add it
            length += (byte & ((1 << 7) - 1)) << bit_range

            # Check MSB and if it is zero, this is the last length byte and we are ready to read string
            if byte & (1 << 7) == 0:
                break
        else:
            # For-else statements in Python are useful! This will be only happen if the for successfully completes
            raise NRBFError('NRBF LengthPrefixedString overflow')

        # Read the string
        return self.stream.read(length).decode('utf-8')

    # endregion

    # region Primitive Array reader functions

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Boolean)
    def _readBoolArray(self, length=1):
        return struct.unpack('<{0}?'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Byte)
    def _readByteArray(self, length=1):
        return struct.unpack('<{0}B'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Double)
    def _readDoubleArray(self, length=1):
        return struct.unpack('<{0}d'.format(length), self.stream.read(8 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int16)
    def _readInt16Array(self, length=1):
        return struct.unpack('<{0}h'.format(length), self.stream.read(2 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int32)
    def _readInt32Array(self, length=1):
        return struct.unpack('<{0}i'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int64)
    def _readInt64Array(self, length=1):
        return struct.unpack('<{0}q'.format(length), self.stream.read(8 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.SByte)
    def _readSByteArray(self, length=1):
        return struct.unpack('<{0}b'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Single)
    def _readSingleArray(self, length=1):
        return struct.unpack('<{0}f'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt16)
    def _readUInt16Array(self, length=1):
        return struct.unpack('<{0}H'.format(length), self.stream.read(2 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt32)
    def _readUInt32Array(self, length=1):
        return struct.unpack('<{0}I'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt64)
    def _readUInt64Array(self, length=1):
        return struct.unpack('<{0}Q'.format(length), self.stream.read(8 * length))

    # endregion

    # region Record reader functions

    @_registerReader(_RecordTypeReaders, 0)
    def _readSerializationHeaderRecord(self):
        print(self._readInt32(), self._readInt32(), self._readInt32())
        print('yay')
        pass
        # self._rootID = self._read_Int32()
        # self._read_Int32()  # HeaderId is ignored
        # major_version = self._read_Int32()
        # minor_version = self._read_Int32()
        # if major_version != 1:
        #     raise NotImplementedError('SerializationHeaderRecord.MajorVersion == {major_version}')
        # if minor_version != 0:
        #     raise NotImplementedError('SerializationHeaderRecord.MinorVersion == {minor_version}')
        # if self._root_id == 0:
        #     raise NotImplementedError('SerializationHeaderRecord.RootId == 0')

    # endregion
