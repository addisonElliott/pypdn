import unittest
import numpy as np
import polarTransform
import sys
import os
from pdn.namedlist import namedlist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestNamedlist(unittest.TestCase):
    def setUp(self):
        self.X = namedlist('X', ['a', 'b', 'c', 'y', 'd'])
        self.Y = namedlist('Y', ['x', 'parent', 'y', 'z'])
        self.Z = namedlist('Z', ['a', 'b', 'c', 'x', 'w'])
        self.W = namedlist('W', ['a', 'b', 'x', 'y', 'c'])
        self.V = namedlist('V', ['a', 'x', 'y', 'z', 'b', 'c'])

        pass

        # Tests to perform on namedlist
        # Check basic _circular_deps to make sure it contains the right items for each thing
        # Check multiple depths, probably like 4-5 to get a good feel
        # Make sure that calling the _check_circular_deps multiple times does nothing
        # Test with and without using slots
        # Test with and without using default values

    def test_function(self):
        x = self.X(10, 'test', 32.0, None, None)
        y = self.Y(x, x, None, None)
        z = self.Z(1, 2, '3', x, None)
        w = self.W(1, 2, x, y, {'this i': 12})
        v = self.V(1, x, y, z, 100, 200)

        x.y = y
        y.y = y
        y.z = z
        z.w = w

        self.assertEqual(x._circular_refs, [])
        self.assertEqual(y._circular_refs, [])
        self.assertEqual(z._circular_refs, [])
        self.assertEqual(w._circular_refs, [])
        self.assertEqual(v._circular_refs, [])

    if __name__ == '__main__':
        unittest.main()
