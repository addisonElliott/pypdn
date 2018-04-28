from enum import IntEnum
import re
from keyword import iskeyword
from pdn.namedlist import namedlist, namedtuple


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

BinaryLibrary = namedlist('BinaryLibrary', ['id', 'name', 'objects'], default=None)
# TODO See about adding more fields to this for resolving!
Reference = namedlist('Reference', ['id'], default=None)
MessageEnd = namedlist('MessageEnd', [])

ObjectNullMultiple = namedtuple('ObjectNullMultiple', 'count')
