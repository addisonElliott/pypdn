# This file should not be tracked by Git.
# It is just meant for basic testing and such to make sure API is working

# from pdn import nrbf
import json
from pdn.namedlist import namedlist, _check_circular_refs
from namedlist import namedtuple

# I want to first begin by testing the nrbf class and cleaning it up so I understand it
# Also, I would like to extend it to WRITE as well, which is not currently the case

# Also, it would be good to test this class with the reading and writing components of it
# Probably nothing too extensive, simple read and check values
# Save those to a new file and check that they are exactly the same.

# filename = 'D:/Users/addis/Desktop/Untitled.pdn'
filename = '../tests/data/imageRawNRBF'
# filename = 'D:/Users/addis/Desktop/Untitled4.pdn'
# filename = 'D:/Users/addis/Desktop/Untitled2.pdn'
# filename = 'D:/Users/addis/Desktop/Untitled3.pdn'

# with open(filename, 'rb') as fh:
#     serial = nrbf.Serialization(fh)
#     data = serial.read_stream()
#
#     # json_encoder = nrbf.JSONEncoder(indent=4, check_circular=True)
#     # print(json_encoder.encode(data))
#     print(data)

# Not having same problem
class PaintDotNet_LayerList2:
    def __init__(self, parent=None):
        self.parent = parent
        self.test = 12
        self.anotherTest = 24

class PaintDotNet_Document2:
    def __init__(self):
        self.layers = PaintDotNet_LayerList2(self)
        self.width = 600
        self.height = 800

xy = PaintDotNet_Document2()

def _repr(self):
    # str = self.__class__.__name__ + '('
    # print(self.__class__.__name__)
    # for name in self._fields:
    #     print('1', name)
    #     str += name + '='
    #     xx = getattr(self, name)
    #     print('2', xx)
    #     print('3', repr(xx))
    #     # str += repr(xx)
    # # ', '.join('{0}={1!r}'.format(name, getattr(self, name)) for name in self._fields)
    # return str

    # return '{0}({1})'.format(self.__class__.__name__,
    #                          ', '.join('{0}={1!r}'.format(name, getattr(self, name)) for name in self._fields))

    if hasattr(self, 'parent_test'):
        y = []
        for name in self._fields:
            if name != self.parent_test:
                y.append('{0}={1!r}'.format(name, getattr(self, name)))

        return '{0}({1})'.format(self.__class__.__name__, ', '.join(y))
    else:
        return '{0}({1})'.format(self.__class__.__name__, ', '.join('{0}={1!r}'.format(name, getattr(self, name)) for name in self._fields))

PaintDotNet_Document = namedlist('PaintDotNet_Document', ['width', 'height', 'id', 'layers'], default=None)
PaintDotNet_LayerList = namedlist('PaintDotNet_LayerList', ['parent', 'test', 'anotherTest', 'id'])

# PaintDotNet_Document = namedlist('PaintDotNet_Document', ['width', 'height', 'layers'], default=None, use_slots=False)
# PaintDotNet_LayerList = namedlist('PaintDotNet_LayerList', ['parent', 'test', 'anotherTest', 'parent_test'], use_slots=False)

x = PaintDotNet_Document(600, 800, 1, None)
y = PaintDotNet_LayerList(x, 12, 24, 2)
x.layers = y
#
# _check_circular_refs(x)
# print(x._circular_refs)
# print(y._circular_refs)
#
print(x)
print(x._asdict())

# Class = namedlist(sanitize_identifier(class_name), member_names, default=None)
# print(x)
# print(dir(x))
# print(x._fields)
# print(xy)
# print(dir(xy))


PaintDotNet_Document = namedlist('PaintDotNet_Document', ['width', 'height', 'layers'], default=None, use_slots=False)

# TODO Make _check_circular_deps function apart of namedlist
# TODO Update repr and asdict to be how I want it to be syntax-wise

# Tests to perform on namedlist
# Check basic _circular_deps to make sure it contains the right items for each thing
# Check multiple depths, probably like 4-5 to get a good feel
# Make sure that calling the _check_circular_deps multiple times does nothing
# Test with and without using slots
# Test with and without using default values