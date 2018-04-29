import os
import sys
import unittest
from pypdn.namedlist import namedlist


class TestNamedlist(unittest.TestCase):
    def setUp(self):
        self.X = namedlist('X', ['id', 'a', 'b', 'c', 'y', 'd'])
        self.Y = namedlist('Y', ['id', 'x', 'parent', 'y', 'z'])
        self.Z = namedlist('Z', ['id', 'a', 'b', 'c', 'x', 'w'])
        self.W = namedlist('W', ['id', 'a', 'b', 'x', 'y', 'c'])
        self.V = namedlist('V', ['id', 'a', 'x', 'y', 'z', 'b', 'c'])

        # Namedlists without slots
        self.Xwos = namedlist('X', ['id', 'a', 'b', 'c', 'y', 'd'], use_slots=False)
        self.Ywos = namedlist('Y', ['id', 'x', 'parent', 'y', 'z'], use_slots=False)
        self.Zwos = namedlist('Z', ['id', 'a', 'b', 'c', 'x', 'w'], use_slots=False)
        self.Wwos = namedlist('W', ['id', 'a', 'b', 'x', 'y', 'c'], use_slots=False)
        self.Vwos = namedlist('V', ['id', 'a', 'x', 'y', 'z', 'b', 'c'], use_slots=False)

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

        self.assertEqual(repr(x), "X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), "
                                  "z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, b=2, x=X(id=1), y=Y(id=2), "
                                  "c={'this i': 12}))), d=None)")

        self.assertEqual(repr(y), "Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), parent=X(id=1, a=10, "
                                  "b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', "
                                  "x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), w=W(id=4, a=1, b=2, x=X(id=1, "
                                  "a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), c={'this i': 12})))")

        self.assertEqual(repr(z), "Z(id=3, a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                  "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), w=W(id=4, a=1, b=2, x=X(id=1, a=10, "
                                  "b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), z=Z(id=3)), "
                                  "d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                  "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3)), "
                                  "c={'this i': 12}))")

        self.assertEquals(repr(w), "W(id=4, a=1, b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4))), "
                                   "d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                   "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, "
                                   "a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                   "w=W(id=4))), c={'this i': 12})")

        self.assertEquals(repr(v), "V(id=5, a=1, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, "
                                   "b=2, x=X(id=1), y=Y(id=2), c={'this i': 12}))), d=None), y=Y(id=2, x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), parent=X(id=1, a=10, b='test', c=32.0, "
                                   "y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), w=W(id=4, a=1, b=2, x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), c={'this i': 12}))), z=Z(id=3, "
                                   "a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), w=W(id=4, a=1, b=2, x=X(id=1, "
                                   "a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), "
                                   "z=Z(id=3)), d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                   "d=None), parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), "
                                   "z=Z(id=3)), c={'this i': 12})), b=100, c=200)")

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

        self.assertEqual(repr(x), "X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), "
                                  "z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, b=2, x=X(id=1), y=Y(id=2), "
                                  "c={'this i': 12}))), d=None)")

        self.assertEqual(repr(y), "Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), parent=X(id=1, a=10, "
                                  "b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', "
                                  "x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), w=W(id=4, a=1, b=2, x=X(id=1, "
                                  "a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), c={'this i': 12})))")

        self.assertEqual(repr(z), "Z(id=3, a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                  "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), w=W(id=4, a=1, b=2, x=X(id=1, a=10, "
                                  "b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), z=Z(id=3)), "
                                  "d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                  "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3)), "
                                  "c={'this i': 12}))")

        self.assertEquals(repr(w), "W(id=4, a=1, b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4))), "
                                   "d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                   "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, "
                                   "a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                   "w=W(id=4))), c={'this i': 12})")

        self.assertEquals(repr(v), "V(id=5, a=1, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, "
                                   "b=2, x=X(id=1), y=Y(id=2), c={'this i': 12}))), d=None), y=Y(id=2, x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), parent=X(id=1, a=10, b='test', c=32.0, "
                                   "y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), w=W(id=4, a=1, b=2, x=X(id=1, a=10, "
                                   "b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), c={'this i': 12}))), z=Z(id=3, "
                                   "a=1, b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                   "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), w=W(id=4, a=1, b=2, x=X(id=1, "
                                   "a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), "
                                   "z=Z(id=3)), d=None), y=Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                   "d=None), parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), "
                                   "z=Z(id=3)), c={'this i': 12})), b=100, c=200)")

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

        print(x._asdict(asString=True))
        print(y._asdict(asString=True))
        print(z._asdict(asString=True))
        print(w._asdict(asString=True))
        print(v._asdict(asString=True))

        self.assertEquals(x._asdict(asString=True), "OrderedDict([('id', 1), ('a', 10), ('b', 'test'), ('c', 32.0), "
                                                    "('y', Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), z=Z(id=3, "
                                                    "a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, b=2, x=X(id=1), "
                                                    "y=Y(id=2), c={'this i': 12})))), ('d', None)])")

        self.assertEquals(y._asdict(asString=True), "OrderedDict([('id', 2), ('x', X(id=1, a=10, b='test', c=32.0, "
                                                    "y=Y(id=2), d=None)), ('parent', X(id=1, a=10, b='test', c=32.0, "
                                                    "y=Y(id=2), d=None)), ('y', Y(id=2)), ('z', Z(id=3, a=1, b=2, "
                                                    "c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "w=W(id=4, a=1, b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                                    "d=None), y=Y(id=2), c={'this i': 12})))])")

        self.assertEquals(z._asdict(asString=True), "OrderedDict([('id', 3), ('a', 1), ('b', 2), ('c', '3'), ('x', "
                                                    "X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                                    "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None)), ('w', W(id=4, "
                                                    "a=1, b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                                    "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), y=Y(id=2, "
                                                    "x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "y=Y(id=2), z=Z(id=3)), c={'this i': 12}))])")

        self.assertEquals(w._asdict(asString=True), "OrderedDict([('id', 4), ('a', 1), ('b', 2), ('x', X(id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), "
                                                    "z=Z(id=3, a=1, b=2, c='3', x=X(id=1), w=W(id=4))), d=None)), "
                                                    "('y', Y(id=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                                    "d=None), parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                                    "d=None), y=Y(id=2), z=Z(id=3, a=1, b=2, c='3', x=X(id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(id=2), d=None), w=W(id=4)))), "
                                                    "('c', {'this i': 12})])")

        self.assertEquals(v._asdict(asString=True), "OrderedDict([('id', 5), ('a', 1), ('x', X(id=1, a=10, b='test', "
                                                    "c=32.0, y=Y(id=2, x=X(id=1), parent=X(id=1), y=Y(id=2), z=Z(id=3, "
                                                    "a=1, b=2, c='3', x=X(id=1), w=W(id=4, a=1, b=2, x=X(id=1), "
                                                    "y=Y(id=2), c={'this i': 12}))), d=None)), ('y', Y(id=2, x=X(id=1, "
                                                    "a=10, b='test', c=32.0, y=Y(id=2), d=None), parent=X(id=1, a=10, "
                                                    "b='test', c=32.0, y=Y(id=2), d=None), y=Y(id=2), z=Z(id=3, a=1, "
                                                    "b=2, c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "w=W(id=4, a=1, b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), "
                                                    "d=None), y=Y(id=2), c={'this i': 12})))), ('z', Z(id=3, a=1, b=2, "
                                                    "c='3', x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                                    "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), w=W(id=4, a=1, "
                                                    "b=2, x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2, x=X(id=1), "
                                                    "parent=X(id=1), y=Y(id=2), z=Z(id=3)), d=None), y=Y(id=2, "
                                                    "x=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "parent=X(id=1, a=10, b='test', c=32.0, y=Y(id=2), d=None), "
                                                    "y=Y(id=2), z=Z(id=3)), c={'this i': 12}))), "
                                                    "('b', 100), ('c', 200)])")


if __name__ == '__main__':
    unittest.main()
