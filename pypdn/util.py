from datetime import datetime, timedelta
import json
import re
from collections import OrderedDict
from decimal import Decimal
from aenum import IntEnum
from keyword import iskeyword
import pypdn.nrbf

from pypdn.namedlist import namedlist, namedtuple


class RecordType(IntEnum):
    SerializedStreamHeader = 0
    ClassWithId = 1
    SystemClassWithMembers = 2
    ClassWithMembers = 3
    SystemClassWithMembersAndTypes = 4
    ClassWithMembersAndTypes = 5
    BinaryObjectString = 6
    BinaryArray = 7
    MemberPrimitiveTyped = 8
    MemberReference = 9
    ObjectNull = 10
    MessageEnd = 11
    BinaryLibrary = 12
    ObjectNullMultiple256 = 13
    ObjectNullMultiple = 14
    ArraySinglePrimitive = 15
    ArraySingleObject = 16
    ArraySingleString = 17
    MethodCall = 21
    MethodReturn = 22


class PrimitiveType(IntEnum):
    Boolean = 1
    Byte = 2
    Char = 3
    Decimal = 5
    Double = 6
    Int16 = 7
    Int32 = 8
    Int64 = 9
    SByte = 10
    Single = 11
    TimeSpan = 12
    DateTime = 13
    UInt16 = 14
    UInt32 = 15
    UInt64 = 16
    Null = 17
    String = 18


class BinaryType(IntEnum):
    Primitive = 0
    String = 1
    Object = 2
    SystemClass = 3
    Class = 4
    ObjectArray = 5
    StringArray = 6
    PrimitiveArray = 7


class BinaryArrayType(IntEnum):
    Single = 0
    Jagged = 1
    Rectangular = 2
    SingleOffset = 3
    JaggedOffset = 4
    RectangularOffset = 5


# Given an identifier string, sanitize the string such that it is suitable to pass to namedlist
def sanitizeIdentifier(identifier):
    # Replace anything that is not an alphanumeric character or underscore with an underscore
    # Also, remove any leading numbers because you cannot reference an member starting with a number
    identifier = re.sub(r'[^a-z0-9_]', '_', identifier, flags=re.IGNORECASE).lstrip('0123456789_')

    # Append an underscore if the identifier is a keyword
    if iskeyword(identifier):
        identifier += '_'

    return identifier


# Take a 1D array and convert it to a N-dimensional array with dimensions given by the arguments dims
# This function uses recursion to accomplish the task so index must be a mutable list with one number inside of it
# The element in the list will be the current index and will be incremented in the recursion
def convert1DArrayND(array1d, dims, index=[0]):
    if len(dims) == 1:
        array = list(array1d[index[0]:index[0] + dims[0]])
        index[0] += dims[0]
        return array
    else:
        return [convert1DArrayND(array1d, dims[1:], index) for x in range(dims[0])]


BinaryLibrary = namedlist('BinaryLibrary', ['_id', 'name', 'objects'], default=None)
Reference = namedlist('Reference', ['_id', 'parent', 'indexInParent', 'resolved', 'collectionResolver', 'originalObj'],
                      default=None)
MessageEnd = namedlist('MessageEnd', [])
ObjectNullMultiple = namedtuple('ObjectNullMultiple', 'count')


# Custom JSONEncoder to convert NRBF class or any of the subclasses into JSON
# This class DOES handle circular references, something that is common in the .NET world
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pypdn.nrbf.NRBF):
            d = OrderedDict(SerializationHeader={'rootID': o.rootID, 'headerID': o.headerID,
                                                 'majorVersion': 1, 'minorVersion': 0})

            # Attach root
            d['Root'] = o.getRoot()

            # Add all binary libraries
            d['BinaryLibraries'] = o.binaryLibraries

            # Attach classes and objects by ID
            d['Objects'] = o.objectsByID

            return d
        elif isinstance(o, Reference):
            # We have to handle the reference specially because there is the parent field that will
            # cause circular dependencies
            return OrderedDict(_class_name=o.__class__.__name__, id=o._id)
        elif hasattr(o, '_asdict'):
            d = OrderedDict(_class_name=o.__class__.__name__)  # prepend the class name
            if hasattr(o, '_id'):
                d['_id'] = o._id
            d.update(o._asdict())
            return d
        elif isinstance(o, (datetime, timedelta)):
            return str(o)
        elif isinstance(o, Decimal):
            return repr(o)

        return super().default(o)

    def afterItem(self, o):
        # pass
        if hasattr(o, '_asdict'):
            o._ref_count = 0
