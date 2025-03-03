# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from itertools import combinations, product

import numpy as np
import pytest

import lontras as lt

from .assertions import assert_array_equal_numpy

example_values = [-1, 0, 1, 2, 3]
example_values_a = [1, 2, 3]
example_values_b = [4, 5, 6]

example_cmp_scalar = 1
example_cmp_array = [example_cmp_scalar] * len(example_values)


class TestArrayInit:
    def test_init_empty(self):
        a = lt.Array([])
        na = np.array([])
        assert_array_equal_numpy(a, na)

    def test_init_integer(self):
        a = lt.Array(example_values)
        na = np.array(example_values)
        assert_array_equal_numpy(a, na)

    def test_init_full(self):
        for i in range(10):
            for j in range(10):
                a = lt.Array.full(i, j)
                na = np.full(i, j)
                assert_array_equal_numpy(a, na)

    def test_init_zeros(self):
        for i in range(10):
            a = lt.Array.zeros(i)
            na = np.zeros(i)
            assert_array_equal_numpy(a, na)

    def test_init_ones(self):
        for i in range(10):
            a = lt.Array.ones(i)
            na = np.ones(i)
            assert_array_equal_numpy(a, na)

    def test__repr__(self):
        a = lt.Array()
        assert str(a) == "Array([])"
        a = lt.Array(example_values)
        assert str(a) == f"Array({example_values!s})"

    def test_shallow_copy(self):
        a = lt.Array([[123]])
        t = a.copy(deep=False)
        a[0][0] = 456
        assert (a == t).all()

    def test_deepcopy(self):
        a = lt.Array([[123]])
        t = a.copy(deep=True)
        a[0][0] = 456
        assert (a != t).all()


class TestArrayGetitem:
    def test_getitem_int(self):
        a = lt.Array(example_values)
        na = np.array(example_values)
        n = len(example_values)
        for i in range(n):
            assert a[i] == na[i]

    def test_getitem_slice(self):
        a = lt.Array(example_values)
        na = np.array(example_values)
        n = len(example_values)
        for i in range(-n, n):
            for j in range(-n, n):
                assert_array_equal_numpy(a[i:j], na[i:j])

    def test_getitem_list(self):
        a = lt.Array(example_values)
        na = np.array(example_values)
        n = len(example_values)
        for i in range(1, n):
            for c in combinations(range(n), r=i):
                indexes = list(c)
                assert_array_equal_numpy(a[indexes], na[indexes])

    def test_getitem_mask(self):
        a = lt.Array(example_values)
        na = np.array(example_values)
        n = len(example_values)
        for mask in product([True, False], repeat=n):
            indexes = list(mask)
            assert_array_equal_numpy(a[indexes], na[indexes])

    def test_getitem_error(self):
        a = lt.Array(example_values)
        with pytest.raises(KeyError, match="Cannot index with: key="):
            a["abc"]


class TestArraySetitem:
    def test_setitem_int(self):
        n = len(example_values)
        for i in range(n):
            a = lt.Array(example_values)
            na = np.array(example_values)
            set_value = i + 99
            a[i] = set_value
            na[i] = set_value
            assert_array_equal_numpy(a, na)

    def test_setitem_slice_to_scalar(self):
        n = len(example_values)
        for i in range(-n, n):
            for j in range(-n, n):
                a = lt.Array(example_values)
                na = np.array(example_values)
                set_value = i + j + 100
                a[i:j] = set_value
                na[i:j] = set_value
                assert_array_equal_numpy(a, na)

    def test_setitem_slice_to_list(self):
        n = len(example_values)
        for i in range(-n, n):
            for j in range(-n, n):
                a = lt.Array(example_values)
                na = np.array(example_values)
                s = slice(i, j)
                set_values = list(range(n))
                a[s] = set_values[s]
                na[s] = set_values[s]
                assert_array_equal_numpy(a, na)

    def test_setitem_list_to_scalar(self):
        n = len(example_values)
        for i in range(1, n):
            for c in combinations(range(n), r=i):
                indexes = list(c)
                a = lt.Array(example_values)
                na = np.array(example_values)
                set_value = i + 100
                a[indexes] = set_value
                na[indexes] = set_value
                assert_array_equal_numpy(a, na)

    def test_setitem_list_to_list(self):
        n = len(example_values)
        for i in range(1, n):
            for c in combinations(range(n), r=i):
                indexes = list(c)
                a = lt.Array(example_values)
                na = np.array(example_values)
                set_values = list(range(len(c)))
                a[indexes] = set_values
                na[indexes] = set_values
                assert_array_equal_numpy(a, na)

    def test_setitem_mask_to_scalar(self):
        n = len(example_values)
        for mask in product([True, False], repeat=n):
            indexes = list(mask)
            a = lt.Array(example_values)
            na = np.array(example_values)
            set_value = 123
            a[indexes] = set_value
            na[indexes] = set_value
            assert_array_equal_numpy(a, na)

    def test_setitem_mask_to_list(self):
        n = len(example_values)
        for mask in product([True, False], repeat=n):
            indexes = list(mask)
            a = lt.Array(example_values)
            na = np.array(example_values)
            set_values = list(range(sum(indexes)))
            a[indexes] = set_values
            na[indexes] = set_values
            assert_array_equal_numpy(a, na)

    def test_setitem_key_error(self):
        a = lt.Array(example_values)
        with pytest.raises(KeyError, match="Cannot index with: key="):
            a["abc"] = 10

    def test_setitem_value_error(self):
        a = lt.Array(example_values)
        with pytest.raises(TypeError, match="Cannot set with: value="):
            a[0] = int

    def test_setitem_invalid_type(self):
        a = lt.Array([1, 2, 3])
        with pytest.raises(KeyError, match="Cannot index with:"):
            a["invalid_index"] = 5


class TestArrayConcatenation:
    def test_append(self):
        a = lt.Array()
        i = []
        a.append(10)
        i.append(10)
        assert (a == i).all()

    def test_append_return(self):
        a = lt.Array()
        i = []
        b = a.append(10)
        i.append(10)
        assert (a == b).all()
        assert (a == i).all()


class TestArrayMapReduce:
    def test_map(self):
        a = lt.Array(example_values)
        assert a.map(lambda x: x**2) == [v**2 for v in example_values]

    def test_reduce(self):
        a = lt.Array(example_values)
        assert a.reduce(lambda acc, cur: acc + cur, 0) == sum(example_values)

    def test_reduce_zero_length(self):
        a = lt.Array()
        assert a.reduce(lambda *_: 10, "did nothing") == "did nothing"

    @pytest.mark.parametrize(
        "func",
        [
            "max",
            "min",
            "sum",
            "all",
            "any",
        ],
    )
    def test_aggregations(self, func):
        a = lt.Array(example_values)
        na = np.array(example_values)
        assert getattr(a, func)() == getattr(na, func)()

    def test_argmax(self):
        m = [0, 10, 20, 0]
        a = lt.Array(m)
        na = np.array(m)
        assert a.argmax() == na.argmax()

    def test_argmin_error(self):
        a = lt.Array()
        with pytest.raises(ValueError, match="Cannot get argmin of an empty Array"):
            a.argmin()

    def test_argmin(self):
        m = [0, 10, 20, 0]
        a = lt.Array(m)
        na = np.array(m)
        assert a.argmin() == na.argmin()

    def test_argmax_error(self):
        a = lt.Array()
        with pytest.raises(ValueError, match="Cannot get argmax of an empty Array"):
            a.argmax()


class TestArrayComparisons:
    @pytest.mark.parametrize(
        "func",
        [
            "__lt__",
            "__le__",
            "__eq__",
            "__ne__",
            "__gt__",
            "__ge__",
        ],
    )
    def test_comparisons(self, func):
        a = lt.Array(example_values)
        na = np.array(example_values)
        # Scalar
        assert_array_equal_numpy(getattr(a, func)(example_cmp_scalar), getattr(na, func)(example_cmp_scalar))
        # ArrayLike
        assert_array_equal_numpy(getattr(a, func)(example_cmp_array), getattr(na, func)(example_cmp_array))


class TestArrayOperators:
    @pytest.mark.parametrize(
        "op",
        [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__pow__",
            "__radd__",
            "__rsub__",
            "__rmul__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rmod__",
            "__rpow__",
        ],
    )
    def test_op(self, op):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        naa = np.array(example_values_a)
        nab = np.array(example_values_b)
        # Scalar
        assert_array_equal_numpy(getattr(aa, op)(example_cmp_scalar), getattr(naa, op)(example_cmp_scalar))
        # Array
        assert_array_equal_numpy(getattr(aa, op)(ab), getattr(naa, op)(nab))
        # ArrayLike
        assert_array_equal_numpy(getattr(aa, op)(example_values_b), getattr(naa, op)(example_values_b))

    def test_matmul(self):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        naa = np.array(example_values_a)
        nab = np.array(example_values_b)

        # Array
        assert ((aa @ ab) == int(naa @ nab)) is True
        # Collection
        assert ((aa @ example_values_b) == int(naa @ example_values_b)) is True
        # Right hand operator
        assert ((example_values_b @ aa) == int(example_values_b @ naa)) is True

    def test_matmul_misaligned_error(self):
        aa = lt.Array(example_values_a)
        ab = lt.Array([*example_values_b, 1])
        with pytest.raises(ValueError, match="Cannot operate Arrays with different sizes:"):
            aa @ ab

    @pytest.mark.parametrize(
        "op",
        [
            "__and__",
            "__xor__",
            "__or__",
            "__rand__",
            "__rxor__",
            "__ror__",
            "__rshift__",
            "__lshift__",
            "__rrshift__",
            "__rlshift__",
        ],
    )
    def test_bop(self, op):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        naa = np.array(example_values_a)
        nab = np.array(example_values_b)
        # Scalar
        assert_array_equal_numpy(getattr(aa, op)(example_cmp_scalar), getattr(naa, op)(example_cmp_scalar))
        # Array
        assert_array_equal_numpy(getattr(aa, op)(ab), getattr(naa, op)(nab))
        # ArrayLike
        assert_array_equal_numpy(getattr(aa, op)(example_values_b), getattr(naa, op)(example_values_b))

    @pytest.mark.parametrize(
        "iop",
        [
            "__iadd__",
            "__isub__",
            "__imul__",
            "__ifloordiv__",
            "__imod__",
            "__ipow__",
            "__iand__",
            "__ixor__",
            "__ior__",
            "__ilshift__",
            "__irshift__",
        ],
    )
    def test_op_inplace(self, iop):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        naa = np.array(example_values_a)
        nab = np.array(example_values_b)
        getattr(aa, iop)(ab)
        getattr(naa, iop)(nab)
        assert_array_equal_numpy(aa, naa)

    def test_op_inplace_truediv(self):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        naa = np.array(example_values_a, dtype=float)
        nab = np.array(example_values_b)
        aa /= ab
        naa /= nab
        assert_array_equal_numpy(aa, naa)

    def test_op_divmod(self):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        d = divmod(aa, ab)
        assert isinstance(d, lt.Array)
        # Numpy expands the dimension of the vector to a matrix. We're just storing the tuples in the vector
        for i, row in enumerate(d):
            assert row == divmod(example_values_a[i], example_values_b[i])

    def test_op_rdivmod(self):
        value = 10
        aa = lt.Array(example_values_a)
        d = divmod(value, aa)
        assert isinstance(d, lt.Array)
        for i, row in enumerate(d):
            assert row == divmod(value, example_values_a[i])

    def test_iop_matmul(self):
        aa = lt.Array(example_values_a)
        ab = lt.Array(example_values_b)
        aa @= ab
        assert aa == np.dot(example_values_a, example_values_b)


class TestArrayUnaryOperators:
    def test_neg(self):
        s = lt.Array(example_values)
        ps = np.array(example_values)
        assert_array_equal_numpy(-s, -ps)

    def test_pos(self):
        s = lt.Array(example_values)
        ps = np.array(example_values)
        assert_array_equal_numpy(+s, +ps)

    def test_abs(self):
        s = lt.Array(example_values)
        ps = np.array(example_values)
        assert_array_equal_numpy(abs(s), abs(ps))

    def test_invert(self):
        s = lt.Array(example_values)
        ps = np.array(example_values)
        assert_array_equal_numpy(~s, ~ps)
