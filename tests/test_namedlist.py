import unittest

from pypdn.namedlist import namedlist


class TestNamedlist(unittest.TestCase):
    def setUp(self):
        self.X = namedlist('X', ['_id', 'a', 'b', 'c', 'y', 'd'])
        self.Y = namedlist('Y', ['_id', 'x', 'parent', 'y', 'z'])
        self.Z = namedlist('Z', ['_id', 'a', 'b', 'c', 'x', 'w'])
        self.W = namedlist('W', ['_id', 'a', 'b', 'x', 'y', 'c'])
        self.V = namedlist('V', ['_id', 'a', 'x', 'y', 'z', 'b', 'c'])

        # Namedlists without slots
        self.Xwos = namedlist('X', ['_id', 'a', 'b', 'c', 'y', 'd'], use_slots=False)
        self.Ywos = namedlist('Y', ['_id', 'x', 'parent', 'y', 'z'], use_slots=False)
        self.Zwos = namedlist('Z', ['_id', 'a', 'b', 'c', 'x', 'w'], use_slots=False)
        self.Wwos = namedlist('W', ['_id', 'a', 'b', 'x', 'y', 'c'], use_slots=False)
        self.Vwos = namedlist('V', ['_id', 'a', 'x', 'y', 'z', 'b', 'c'], use_slots=False)

    def test_repr_basic(self):
        x = self.X(1, 10, 'test', 32.0, None, None)
        y = self.Y(2, x, x, None, None)
        z = self.Z(3, 1, 2, '3', x, None)
        w = self.W(4, 1, 2, x, y, {'this i': 12})
        v = self.V(5, 1, x, y, z, 100, 200)

        x.y = y
        y.y = y
        y.z = z
        z.w = w

        self.assertEqual(x._ref_count, 0)
        self.assertEqual(y._ref_count, 0)
        self.assertEqual(z._ref_count, 0)
        self.assertEqual(w._ref_count, 0)
        self.assertEqual(v._ref_count, 0)

        self.assertEqual(repr(x), "X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), "
                                  "y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, a=1, b=2, x=X(_id=1),"
                                  " y=Y(_id=2), c={'this i': 12}))), d=None)")

        self.assertEqual(repr(y), "Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), parent=X(_id=1, "
                                  "a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), z=Z(_id=3, a=1, b=2, "
                                  "c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), w=W(_id=4, a=1, b=2, "
                                  "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), c={'this i': "
                                  "12})))")

        self.assertEqual(repr(z), "Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                  "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), w=W(_id=4, a=1, b=2, x=X(_id=1, "
                                  "a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), y=Y(_id=2), "
                                  "z=Z(_id=3)), d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                  "d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), "
                                  "z=Z(_id=3)), c={'this i': 12}))")

        self.assertEqual(repr(w), "W(_id=4, a=1, b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                   "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4))), "
                                   "d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                   "parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), "
                                   "z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                   "d=None), w=W(_id=4))), c={'this i': 12})")

        self.assertEqual(repr(v), "V(_id=5, a=1, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                   "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, "
                                   "a=1, b=2, x=X(_id=1), y=Y(_id=2), c={'this i': 12}))), d=None), y=Y(_id=2, "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), parent=X(_id=1, a=10, "
                                   "b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), w=W(_id=4, a=1, b=2, "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), c={'this i': "
                                   "12}))), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, "
                                   "x=X(_id=1), parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), w=W(_id=4, a=1, "
                                   "b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), "
                                   "y=Y(_id=2), z=Z(_id=3)), d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, "
                                   "y=Y(_id=2), d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                   "y=Y(_id=2), z=Z(_id=3)), c={'this i': 12})), b=100, c=200)")

    def test_repr_without_slots(self):
        x = self.Xwos(1, 10, 'test', 32.0, None, None)
        y = self.Ywos(2, x, x, None, None)
        z = self.Zwos(3, 1, 2, '3', x, None)
        w = self.Wwos(4, 1, 2, x, y, {'this i': 12})
        v = self.Vwos(5, 1, x, y, z, 100, 200)

        x.y = y
        y.y = y
        y.z = z
        z.w = w

        self.assertEqual(x._ref_count, 0)
        self.assertEqual(y._ref_count, 0)
        self.assertEqual(z._ref_count, 0)
        self.assertEqual(w._ref_count, 0)
        self.assertEqual(v._ref_count, 0)

        self.assertEqual(repr(x),
                         "X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), y=Y(_id=2), "
                         "z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, a=1, b=2, x=X(_id=1), y=Y(_id=2), "
                         "c={'this i': 12}))), d=None)")

        self.assertEqual(repr(y),
                         "Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), parent=X(_id=1, a=10, "
                         "b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', "
                         "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), w=W(_id=4, a=1, b=2, x=X(_id=1, "
                         "a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), c={'this i': 12})))")

        self.assertEqual(repr(z), "Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                  "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), w=W(_id=4, a=1, b=2, x=X(_id=1, "
                                  "a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), y=Y(_id=2), "
                                  "z=Z(_id=3)), d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                  "d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), "
                                  "z=Z(_id=3)), c={'this i': 12}))")

        self.assertEqual(repr(w), "W(_id=4, a=1, b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                   "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4))), "
                                   "d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                   "parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), "
                                   "z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                   "d=None), w=W(_id=4))), c={'this i': 12})")

        self.assertEqual(repr(v), "V(_id=5, a=1, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                   "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, "
                                   "a=1, b=2, x=X(_id=1), y=Y(_id=2), c={'this i': 12}))), d=None), y=Y(_id=2, "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), parent=X(_id=1, a=10, "
                                   "b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), w=W(_id=4, a=1, b=2, "
                                   "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), c={'this i': "
                                   "12}))), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, "
                                   "x=X(_id=1), parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), w=W(_id=4, a=1, "
                                   "b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), "
                                   "y=Y(_id=2), z=Z(_id=3)), d=None), y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, "
                                   "y=Y(_id=2), d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                   "y=Y(_id=2), z=Z(_id=3)), c={'this i': 12})), b=100, c=200)")

    def test_asdict_basic(self):
        x = self.X(1, 10, 'test', 32.0, None, None)
        y = self.Y(2, x, x, None, None)
        z = self.Z(3, 1, 2, '3', x, None)
        w = self.W(4, 1, 2, x, y, {'this i': 12})
        v = self.V(5, 1, x, y, z, 100, 200)

        x.y = y
        y.y = y
        y.z = z
        z.w = w

        self.assertEqual(x._ref_count, 0)
        self.assertEqual(y._ref_count, 0)
        self.assertEqual(z._ref_count, 0)
        self.assertEqual(w._ref_count, 0)
        self.assertEqual(v._ref_count, 0)

        self.assertEqual(x._asdict(asString=True), "OrderedDict([('_id', 1), ('a', 10), ('b', 'test'), ('c', 32.0), "
                                                    "('y', Y(_id=2, x=X(_id=1), parent=X(_id=1), y=Y(_id=2), "
                                                    "z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, a=1, b=2, "
                                                    "x=X(_id=1), y=Y(_id=2), c={'this i': 12})))), ('d', None)])")

        self.assertEqual(y._asdict(asString=True), "OrderedDict([('_id', 2), ('x', X(_id=1, a=10, b='test', c=32.0, "
                                                    "y=Y(_id=2), d=None)), ('parent', X(_id=1, a=10, b='test', c=32.0, "
                                                    "y=Y(_id=2), d=None)), ('y', Y(_id=2)), ('z', Z(_id=3, a=1, b=2, "
                                                    "c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                                    "w=W(_id=4, a=1, b=2, x=X(_id=1, a=10, b='test', c=32.0, "
                                                    "y=Y(_id=2), d=None), y=Y(_id=2), c={'this i': 12})))])")

        self.assertEqual(z._asdict(asString=True), "OrderedDict([('_id', 3), ('a', 1), ('b', 2), ('c', '3'), ('x', "
                                                    "X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                                    "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None)), ('w', "
                                                    "W(_id=4, a=1, b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, "
                                                    "x=X(_id=1), parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), "
                                                    "y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                                    "d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                                    "d=None), y=Y(_id=2), z=Z(_id=3)), c={'this i': 12}))])")

        self.assertEqual(w._asdict(asString=True), "OrderedDict([('_id', 4), ('a', 1), ('b', 2), ('x', X(_id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), "
                                                    "y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), "
                                                    "w=W(_id=4))), d=None)), ('y', Y(_id=2, x=X(_id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(_id=2), d=None), parent=X(_id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(_id=2), d=None), y=Y(_id=2), z=Z(_id=3, "
                                                    "a=1, b=2, c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                                    "d=None), w=W(_id=4)))), ('c', {'this i': 12})])")

        self.assertEqual(v._asdict(asString=True), "OrderedDict([('_id', 5), ('a', 1), ('x', X(_id=1, a=10, b='test', "
                                                    "c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), y=Y(_id=2), "
                                                    "z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, a=1, b=2, "
                                                    "x=X(_id=1), y=Y(_id=2), c={'this i': 12}))), d=None)), ('y', "
                                                    "Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                                    "parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                                    "y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(_id=2), d=None), w=W(_id=4, a=1, b=2, "
                                                    "x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), d=None), "
                                                    "y=Y(_id=2), c={'this i': 12})))), ('z', Z(_id=3, a=1, b=2, "
                                                    "c='3', x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), "
                                                    "parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), w=W(_id=4, "
                                                    "a=1, b=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, "
                                                    "x=X(_id=1), parent=X(_id=1), y=Y(_id=2), z=Z(_id=3)), d=None), "
                                                    "y=Y(_id=2, x=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                                    "d=None), parent=X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2), "
                                                    "d=None), y=Y(_id=2), z=Z(_id=3)), c={'this i': 12}))), ('b', "
                                                    "100), ('c', 200)])")

    def test_repr_basic_py3_8(self):
        """ test_repr_basic on python 3.8+

        Set up the named lists. These will work with the patch,
        without the patch these will throw a type error

        see https://github.com/addisonElliott/pypdn/issues/3 for details
        of the error trace

        Then check that the repr of the object is as in previous versions
        of python
        """

        # Set up the named lists here
        testX = namedlist('X', ['_id', 'a', 'b', 'c', 'y', 'd'])
        testY = namedlist('Y', ['_id', 'x', 'parent', 'y', 'z'])
        testZ = namedlist('Z', ['_id', 'a', 'b', 'c', 'x', 'w'])
        testW = namedlist('W', ['_id', 'a', 'b', 'x', 'y', 'c'])
        testV = namedlist('V', ['_id', 'a', 'x', 'y', 'z', 'b', 'c'])

        # Mirror a subset of test_repr_basic to test that x is identical to
        # before python 3.8
        x = testX(1, 10, 'test', 32.0, None, None)
        y = testY(2, x, x, None, None)
        z = testZ(3, 1, 2, '3', x, None)
        w = testW(4, 1, 2, x, y, {'this i': 12})
        v = testV(5, 1, x, y, z, 100, 200)

        x.y = y
        y.y = y
        y.z = z
        z.w = w
        self.assertEqual(x._ref_count, 0)

        self.assertEqual(repr(x), "X(_id=1, a=10, b='test', c=32.0, y=Y(_id=2, x=X(_id=1), parent=X(_id=1), "
                                  "y=Y(_id=2), z=Z(_id=3, a=1, b=2, c='3', x=X(_id=1), w=W(_id=4, a=1, b=2, x=X(_id=1),"
                                  " y=Y(_id=2), c={'this i': 12}))), d=None)")

if __name__ == '__main__':
    unittest.main()
