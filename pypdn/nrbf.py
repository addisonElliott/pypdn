import struct
from datetime import timezone
from functools import reduce

from pypdn.util import *


# This library was inspired from the following code:
#   https://github.com/gurnec/Undo_FFG/blob/master/nrbf.py
# Thanks to Christopher Gurnee!

# TODO Do documentation and docstrings for this
# TODO Issue with reporting _id in JSON or something. classID vs _id is issue

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

    # Dictionary contains stuff for reading primitive arrays
    _PrimitiveTypeArrayReaders = {}

    # Dictionary containing readers for additional info in classes
    _AdditionalInfoReaders = {}

    def __init__(self, stream=None, filename=None):
        self.stream = None
        self.rootID = None
        self.headerID = None
        self.referencesResolved = None

        # Dictionary of binary libraries with their linked objects
        self.binaryLibraries = {}

        # Keep track of class and objects by their ID
        self.classByID = {}
        self.objectsByID = {}

        # Keeps track of references so that after reading is done, the references can be resolved
        # collection references replace the system collections with Python equivalent types such as dict or list
        self._collectionReferences = []

        # If a stream or filename to be loaded is given, then call the read function
        # This makes the syntax cleaner to allow this creation of class and loading of data all in one line
        if stream is not None or filename is not None:
            self.read(stream, filename)

    def read(self, stream=None, filename=None):
        if stream is None and filename is not None:
            stream = open(filename, 'rb')

        assert stream.readable()

        if self.stream is not None:
            # Means we are in the middle of reading or writing data, throw exception probably
            raise NRBFError('Class is already reading from a stream! Please close the stream before trying again')

        if self.rootID is not None:
            # File has already been loaded so we must reset everything
            self.rootID = None
            self.headerID = None
            self.binaryLibraries = {}
            self.classByID = {}
            self.objectsByID = {}
            self._collectionReferences = []

        self.stream = stream

        # Read header
        self._readRecord()
        if self.rootID is None:
            raise NRBFError('Invalid stream, unable to read header. File may be corrupted')

        # Keep reading records until we receive a MessageEnd record
        while not isinstance(self._readRecord(), MessageEnd):
            pass

        # Resolve all the collection references
        self.resolveReferences()

        # Once we are done reading, we set the stream to None because it will not be used
        self.stream = None

    def write(self, stream=None, filename=None):
        if stream is None and filename is not None:
            stream = open(filename, 'wb')

        assert stream.writable()
        assert stream.seekable()

        # TODO Write NRBF files
        raise NotImplementedError('Writing a NRBF file is not supported yet')

    def getRoot(self):
        assert self.rootID is not None

        return self.objectsByID[self.rootID]

    def resolveReferences(self):
        # Resolve all the collection references
        for reference in self._collectionReferences:
            # Calls one of the resolvers
            replacement = reference.collectionResolver(self, reference)

            # The final steps common to all collection resolvers are completed below
            if reference.parent:
                reference.parent[reference.index_in_parent] = replacement

            self.objectsByID[reference._id] = replacement

        # Note: Collection references must be saved so that it can be converted back
        # when saving the file again.

        # Loop through all of the objects
        for _, object in self.objectsByID.items():
            # Attempt to iterate through the object
            # If it doesnt work, then move on to the next item because there wont need to be any
            # references.
            # Otherwise, if any of the items have an object ID, create a reference
            try:
                for index, item in enumerate(object):
                    if isinstance(item, Reference):
                        self._resolveSimpleReference(item)
            except TypeError:
                pass

        self.referencesResolved = True

    def unresolveReferences(self):
        # Loop through all of the objects
        for _, object in self.objectsByID.items():
            # Attempt to iterate through the object
            # If it doesnt work, then move on to the next item because there wont need to be any
            # references.
            # Otherwise, if any of the items have an object ID, create a reference
            try:
                for index, item in enumerate(object):
                    if hasattr(item, '_id'):
                        object[index] = Reference(item._id, object, index)
            except TypeError:
                pass

        self.referencesResolved = False

    def toJSON(self, resolveReferences=True, **kwargs):
        # Resolve or unresolve the references based on what the user desires
        # and the current state
        if resolveReferences and not self.referencesResolved:
            self.resolveReferences()
        elif not resolveReferences and self.referencesResolved:
            self.unresolveReferences()

        jsonEncoder = JSONEncoder(**kwargs)
        return jsonEncoder.encode(self)

    # region Primitive reader functions

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Boolean, _PrimitiveTypeStructs, '?')
    def _readBool(self):
        return self._PrimitiveTypeStructs[PrimitiveType.Boolean].unpack(self.stream.read(1))[0]

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.Byte, _PrimitiveTypeStructs, 'B')
    @_registerReader(_AdditionalInfoReaders, BinaryType.Primitive)
    @_registerReader(_AdditionalInfoReaders, BinaryType.PrimitiveArray)
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
        return timedelta(microseconds=self._readInt64() / 10)

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
        time = datetime(1, 1, 1)
        try:
            time += timedelta(microseconds=ticks / 10)
        except OverflowError:
            pass

        # Update datetime object to have the appropriate timezone
        # If kind is 1, then this is UTC and if kind is 2, then this is local timezone
        if kind == 1:
            time = time.replace(tzinfo=timezone.utc)
        elif kind == 2:
            LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo
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
    @_registerReader(_AdditionalInfoReaders, BinaryType.String)
    @_registerReader(_AdditionalInfoReaders, BinaryType.Object)
    @_registerReader(_AdditionalInfoReaders, BinaryType.ObjectArray)
    @_registerReader(_AdditionalInfoReaders, BinaryType.StringArray)
    @_registerReader(_RecordTypeReaders, RecordType.ObjectNull)
    def _readNull(self):
        return None

    @_registerReader(_PrimitiveTypeReaders, PrimitiveType.String)
    @_registerReader(_AdditionalInfoReaders, BinaryType.SystemClass)
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
    def _readBoolArray(self, length):
        return struct.unpack('<{0}?'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Byte)
    def _readByteArray(self, length):
        return struct.unpack('<{0}B'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Double)
    def _readDoubleArray(self, length):
        return struct.unpack('<{0}d'.format(length), self.stream.read(8 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int16)
    def _readInt16Array(self, length):
        return struct.unpack('<{0}h'.format(length), self.stream.read(2 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int32)
    def _readInt32Array(self, length):
        return struct.unpack('<{0}i'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Int64)
    def _readInt64Array(self, length):
        return struct.unpack('<{0}q'.format(length), self.stream.read(8 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.SByte)
    def _readSByteArray(self, length):
        return struct.unpack('<{0}b'.format(length), self.stream.read(length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.Single)
    def _readSingleArray(self, length):
        return struct.unpack('<{0}f'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt16)
    def _readUInt16Array(self, length):
        return struct.unpack('<{0}H'.format(length), self.stream.read(2 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt32)
    def _readUInt32Array(self, length):
        return struct.unpack('<{0}I'.format(length), self.stream.read(4 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.UInt64)
    def _readUInt64Array(self, length):
        return struct.unpack('<{0}Q'.format(length), self.stream.read(8 * length))

    @_registerReader(_PrimitiveTypeArrayReaders, PrimitiveType.String)
    def _readStringArray(self, length):
        return [self._readString() for i in range(length)]

    # endregion

    # region AdditionalInfo reader functions

    # Note: Remaining reader functions are attached in other sections to existing functions

    # The ClassTypeInfo structure is read and ignored
    @_registerReader(_AdditionalInfoReaders, BinaryType.Class)
    def _readClassTypeInfo(self):
        name = self._readString()
        libraryID = self._readInt32()

        return (name, libraryID)

    # endregion

    # region Record reader functions

    def _readRecord(self):
        recordType = self._readByte()

        return self._RecordTypeReaders[recordType](self)

    def _readPrimitive(self, primitiveType):
        return self._PrimitiveTypeReaders[primitiveType](self)

    @_registerReader(_RecordTypeReaders, RecordType.SerializedStreamHeader)
    def _readSerializationHeaderRecord(self):
        self.rootID = self._readInt32()
        self.headerID = self._readInt32()
        majorVersion, minorVersion = self._readInt32(), self._readInt32()

        if majorVersion != 1 or minorVersion != 0:
            raise NRBFError('Major and minor version for serialization header is incorrect: {0} {1}'.format(
                majorVersion, minorVersion))

        if self.rootID == 0:
            raise NotImplementedError(
                'Root ID is zero indicating that a BinaryMethodCall is available. Not implemented yet')

    @_registerReader(_RecordTypeReaders, RecordType.ClassWithId)
    def _readClassWithId(self):
        objectID = self._readInt32()
        metadataID = self._readInt32()
        cls = self.classByID[metadataID]

        # Only instance where the objectID does NOT equal the class ID is here!
        return self._readClassMembers(cls(), objectID)

    @_registerReader(_RecordTypeReaders, RecordType.SystemClassWithMembers)
    def _readSystemClassWithMembers(self):
        cls = self._readClassInfo(isSystemClass=True)
        cls._typeInfo = None

        return self._readClassMembers(cls(), cls._id)

    @_registerReader(_RecordTypeReaders, RecordType.ClassWithMembers)
    def _readClassWithMembers(self):
        cls = self._readClassInfo(isSystemClass=False)
        cls._typeInfo = None
        libraryID = self._readInt32()

        return self._readClassMembers(cls(), cls._id, libraryID)

    @_registerReader(_RecordTypeReaders, RecordType.SystemClassWithMembersAndTypes)
    def _readSystemClassWithMembersAndTypes(self):
        cls = self._readClassInfo(isSystemClass=True)
        self._readMemberTypeInfo(cls)

        return self._readClassMembers(cls(), cls._id)

    @_registerReader(_RecordTypeReaders, RecordType.ClassWithMembersAndTypes)
    def _readClassWithMembersAndTypes(self):
        cls = self._readClassInfo(isSystemClass=False)
        self._readMemberTypeInfo(cls)
        libraryID = self._readInt32()

        return self._readClassMembers(cls(), cls._id, libraryID)

    @_registerReader(_RecordTypeReaders, RecordType.BinaryObjectString)
    def _readBinaryObjectString(self):
        objectID = self._readInt32()
        string = self._readString()
        self.objectsByID[objectID] = string

        return string

    @_registerReader(_RecordTypeReaders, RecordType.MemberPrimitiveTyped)
    def _readMemberPrimitiveTyped(self):
        primitiveType = self._readByte()
        value = self._PrimitiveTypeReaders[primitiveType](self)

        return value

    @_registerReader(_RecordTypeReaders, RecordType.BinaryArray)
    def _readBinaryArray(self):
        objectID = self._readInt32()
        arrayType = self._readByte()
        rank = self._readInt32()
        lengths = [self._readInt32() for i in range(rank)]

        # The lower bounds are ignored currently
        # Not sure of the implications or purpose of this
        if arrayType in [BinaryArrayType.SingleOffset, BinaryArrayType.JaggedOffset, BinaryArrayType.RectangularOffset]:
            lowerBounds = [self._readInt32() for i in range(rank)]

        binaryType = self._readByte()
        additionalInfo = self._AdditionalInfoReaders[binaryType](self)

        # Get total length of items that we need to read
        # This is just the product of all the elements in the lengths array
        length = reduce(lambda x, y: x * y, lengths)

        # If the items are primitives, use primitive array readers
        # Otherwise, the items will be objects and should be read by reading records
        if binaryType == BinaryType.Primitive:
            array = self._PrimitiveTypeArrayReaders[additionalInfo](self, length)
        else:
            array = self._readObjectArray(length, objectID)

        # For a multidimensional array, take the 1D array that was read and convert it to ND
        if arrayType in [BinaryArrayType.Rectangular, BinaryArrayType.RectangularOffset]:
            array = convert1DArrayND(array, lengths)

        # Save the object by ID
        # Only required for primitive because _readObjectArray saves the ID for you
        # But we just overwrite it regardless
        self.objectsByID[objectID] = array

        return array

    # When object's with an object ID are encountered above, they are added to the _objectsByID dictionary.
    # A MemberReference object contains the object ID that the reference refers to. These references are
    # resolved at the end.
    @_registerReader(_RecordTypeReaders, RecordType.MemberReference)
    def _readMemberReference(self):
        # objectID
        ref = Reference(self._readInt32())

        return ref

    @_registerReader(_RecordTypeReaders, RecordType.MessageEnd)
    def _readMessageEnd(self):
        return MessageEnd()

    @_registerReader(_RecordTypeReaders, RecordType.BinaryLibrary)
    def _readBinaryLibrary(self):
        libraryID = self._readInt32()
        libraryName = self._readString()
        library = BinaryLibrary(libraryID, libraryName, {})
        self.binaryLibraries[libraryID] = library

        return library

    @_registerReader(_RecordTypeReaders, RecordType.ObjectNullMultiple256)
    def _readObjectNullMultiple256(self):
        # Count
        return ObjectNullMultiple(self._readByte())

    @_registerReader(_RecordTypeReaders, RecordType.ObjectNullMultiple)
    def _readObjectNullMultiple(self):
        # Count
        return ObjectNullMultiple(self._readInt32())

    @_registerReader(_RecordTypeReaders, RecordType.ArraySinglePrimitive)
    def _readArraySinglePrimitive(self):
        objectID, length = self._readArrayInfo()
        primitiveType = self._readByte()

        array = self._PrimitiveTypeArrayReaders[primitiveType](self, length)
        self.objectsByID[objectID] = array

        return array

    @_registerReader(_RecordTypeReaders, RecordType.ArraySingleObject)
    @_registerReader(_RecordTypeReaders, RecordType.ArraySingleString)
    def _read_ArraySingleObject(self):
        objectID, length = self._readArrayInfo()

        array = self._readObjectArray(length, objectID)
        self.objectsByID[objectID] = array

        return array

    @_registerReader(_RecordTypeReaders, RecordType.MethodCall)
    def _read_MethodCall(self):
        raise NotImplementedError('MethodCall')

    @_registerReader(_RecordTypeReaders, RecordType.MethodReturn)
    def _read_MethodReturn(self):
        raise NotImplementedError('MethodReturn')

    # endregion

    # region Read helper classes

    def _readArrayInfo(self):
        # objectID and array length
        return (self._readInt32(), self._readInt32())

    # Reads a ClassInfo structure and creates and saves a new namedlist object with the same member and ClassInfo
    # specifics
    def _readClassInfo(self, isSystemClass=False):
        objectID = self._readInt32()
        className = self._readString()
        memberCount = self._readInt32()
        memberNames = [sanitizeIdentifier(self._readString()) for i in range(memberCount)]

        # Create namedlist that will act as the class
        # Set object ID for the class and save it by the object ID for later
        cls = namedlist(sanitizeIdentifier(className), memberNames, default=None)

        # Class ID is a special identifier to represent the class itself but not the data within of it
        # For instance, you can declare the class once but then instantiate it multiple times
        # The object ID is the unique identifier for the object itself and cannot be repeated
        # In this instance, the classID and objectID are the same for the first instantiation
        # The only way to get these different is to have a ClassWithID record and then this is
        # set manually
        cls._classID = cls._id = objectID
        cls._isSystemClass = isSystemClass
        self.classByID[objectID] = cls

        return cls

    def _readMemberTypeInfo(self, cls):
        binaryTypes = [self._readByte() for member in cls._fields]
        additionalInfo = [self._AdditionalInfoReaders[type](self) for type in binaryTypes]

        # Combine the binary types and additional info into one tuple
        # This gets saved to the class object because it will be used to set the members
        # Also, if there is a ClassWithId object then it won't have type information but only
        # the class ID so we will need to retain this information.
        cls._typeInfo = tuple(zip(binaryTypes, additionalInfo))

    # Reads members or array elements into the 'obj' pre-allocated list or class instance
    def _readClassMembers(self, obj, objectID, libraryID=None):
        index = 0
        while index < len(obj._fields):
            # If typeinfo is not defined, as is the case for ClassWithMembers and SystemClassWithMembers,
            # then assume it is an object that can be read
            # Not sure if this is a safe assumption because Microsoft isn't entirely clear when the member type
            # information is 'unnecessary'
            if obj._typeInfo is None:
                binaryType, additionalInfo = BinaryType.Object, None
            else:
                binaryType, additionalInfo = obj._typeInfo[index]

            if binaryType == BinaryType.Primitive:
                value = self._readPrimitive(additionalInfo)
            else:
                value = self._readRecord()

                if isinstance(value, BinaryLibrary):
                    # BinaryLibrary can precede the actual member
                    # Continue on to the actual member
                    continue
                elif isinstance(value, ObjectNullMultiple):
                    # Skip a given number of indices
                    index += value.count
                    continue
                elif isinstance(value, Reference):
                    value.parent = obj
                    value.indexInParent = index

            obj[index] = value
            index += 1

        # If this object is a .NET collection (e.g. a Generic dict or list) which can be
        # replaced by a native Python type, insert a collection _Reference instead of the raw
        # object which will be resolved later in read() using a "collection resolver"
        if getattr(obj.__class__, '_isSystemClass', False):
            for name, resolver in self._collectionResolvers:
                if obj.__class__.__name__.startswith('System_Collections_Generic_%s_' % name):
                    obj = Reference(objectID, collectionResolver=resolver, originalObj=obj)
                    self._collectionReferences.append(obj)
                    break

        self.objectsByID[objectID] = obj

        # Use libraryID to append the class to that binary library
        # This is particularly useful when saving the items again so that you save all of a binary library at once
        if libraryID:
            self.binaryLibraries[libraryID].objects[objectID] = obj

        return obj

    def _readObjectArray(self, length, objectID):
        array = [None] * length

        index = 0
        while index < length:
            value = self._readRecord()

            if isinstance(value, BinaryLibrary):
                # BinaryLibrary can precede the actual member
                # Continue on to the actual member
                continue
            elif isinstance(value, ObjectNullMultiple):
                # Skip a given number of indices
                index += value.count
                continue
            elif isinstance(value, Reference):
                value.parent = array
                value.indexInParent = index

            array[index] = value
            index += 1

        return array

    # endregion

    # region Resolve references functions

    # Convert a _Reference representing a MemberReference into its referenced object
    def _resolveSimpleReference(self, reference):
        if reference.resolved:
            return

        replacement = self.objectsByID[reference._id]
        reference.parent[reference.indexInParent] = replacement

        # Not sure when this would happen, doesn't hurt though!
        reference.resolved = True

    # Convert a _Reference representing a .NET dictionary collection into a Python dict
    def _resolveDictReference(self, reference):
        originalObj = reference.originalObj

        # If the key-value pairs of the dict are itself a Reference, then resolve those first
        if isinstance(originalObj.KeyValuePairs, Reference):
            self._resolveSimpleReference(originalObj.KeyValuePairs)

        replacement = {}

        for item in originalObj.KeyValuePairs:
            try:
                # If any key is a _Reference, it must be resolved first
                # (value _References will be resolved later)
                if isinstance(item.key, Reference):
                    self._resolveSimpleReference(item.key)

                assert item.key not in replacement
                replacement[item.key] = item.value
            except (AssertionError, TypeError):
                # Not all .NET dictionaries can be converted to Python dicts
                # If the conversion fails, just proceed w/ the original object
                replacement = originalObj
                break
        else:
            # Assuming that the for loop completed successfully, indicating that all
            # of the dictionary objects were converted,
            # Then we need to fix the dictionary values for References
            for key, value in replacement.items():
                if isinstance(value, Reference):
                    value.parent = replacement
                    value.index_in_parent = key

        return replacement

    # Convert a Reference representing a .NET list collection into a Python list
    def _resolveListReference(self, reference):
        originalObj = reference.originalObj

        # If the components of the list are itself a Reference, then resolve those first
        if isinstance(originalObj.items, Reference):
            self._resolveSimpleReference(originalObj.items)

        replacement = originalObj.items

        # Update parent for all replacement elements if they are references
        for element in replacement:
            if isinstance(element, Reference):
                element.parent = replacement

        return replacement

    _collectionResolvers = (
        ('Dictionary', _resolveDictReference),
        ('List', _resolveListReference)
    )

    # endregion
