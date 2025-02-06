# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT

import statistics
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import lontras as lt

from .assertions import assert_exception, assert_series_equal_pandas

example_dict = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_index = tuple(example_dict.keys())
example_values = tuple(example_dict.values())
example_dict_a = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_dict_b = MappingProxyType({"a": 4, "b": 5, "c": 6})
example_dict_no_keys = MappingProxyType({0: 1, 1: 2, 2: 3})
example_scalar = 3
example_name = "snake"
example_dict_series_str = """a  1
b  2
c  3
name: None
"""
example_dict_series_with_name_str = """a  1
b  2
c  3
name: snake
"""
example_stats = [0, 1, 2, 3, 4, 5, 6, 6]
example_unary = [-3, -1, 0, 1, 2]


class TestSeriesInit:
    def test_init_empty(self):
        s = lt.Series()
        ps = pd.Series()
        assert_series_equal_pandas(s, ps)

    def test_init_mapping(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_series_equal_pandas(s, ps)

    def test_init_collection_with_index(self):
        s = lt.Series(example_dict.values(), index=example_index)
        ps = pd.Series(example_dict.values(), index=example_index)
        assert_series_equal_pandas(s, ps)

    def test_init_collection(self):
        s = lt.Series(tuple(example_dict.values()))
        ps = pd.Series(tuple(example_dict.values()))
        assert_series_equal_pandas(s, ps)

    def test_init_scalar(self):
        s = lt.Series(0)
        ps = pd.Series(0)
        assert_series_equal_pandas(s, ps)

    def test_init_mapping_with_index(self):
        s = lt.Series(example_dict, index=example_index)
        ps = pd.Series(example_dict, index=example_index)
        assert_series_equal_pandas(s, ps)

    def test_init_error_index_mismatch(self):
        assert_exception(
            lambda: pd.Series([0, 1, 2], index=[0, 1]), lambda: lt.Series([0, 1, 2], index=[0, 1]), ValueError
        )

    def test_init_error_unexpected_data_type(self):
        with pytest.raises(ValueError, match="Unexpected data type:"):
            lt.Series(int)

    def test__str__(self):
        s = lt.Series()
        assert str(s) == "Empty Series"
        s = lt.Series(name=example_name)
        assert str(s) == f'Empty Series(name="{example_name}")'
        s = lt.Series(example_dict)
        assert str(s) == example_dict_series_str
        s = lt.Series(example_dict, name=example_name)
        assert str(s) == example_dict_series_with_name_str

    def test_name(self):
        s = lt.Series(0, name=example_name)
        ps = pd.Series(0, name=example_name)
        assert_series_equal_pandas(s, ps)

    def test_rename(self):
        s = lt.Series(0, name=example_name)
        ps = pd.Series(0, name=example_name)
        new_name = "cobra"
        s.rename(new_name)  # Should not mutate
        assert_series_equal_pandas(s, ps)
        ps.rename(new_name)
        assert_series_equal_pandas(s.rename(new_name), ps.rename(new_name))
        assert_series_equal_pandas(s, ps)

    def test_shallow_copy(self):
        s = lt.Series([[123]])
        ps = pd.Series([[123]])
        t = s.copy(deep=False)
        pt = ps.copy(deep=False)
        s.iloc[0][0] = [456]
        ps.iloc[0][0] = [456]
        assert_series_equal_pandas(t, pt)
        assert_series_equal_pandas(s, ps)

    def test_deepcopy(self):
        s = lt.Series([[123]])
        p = s.copy(deep=False)
        q = s.copy(deep=True)
        s.iloc[0][0] = [456]
        # Pandas does not copy objects recursively
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.copy.html
        assert (s == p).all()
        assert (s != q).all()

    def test_index_getter(self):
        s = lt.Series(example_dict)
        assert s.index == list(example_index)

    def test_index_setter(self):
        s = lt.Series(example_dict)
        s.index = list(reversed(example_index))
        assert s.index == list(reversed(example_index))

    def test_reindex(self):
        s = lt.Series(example_dict)
        s.reindex(list(reversed(example_index)))  # Should not mutate
        assert s.index == list(example_index)
        s = s.reindex(list(reversed(example_index)))
        assert s.index == list(reversed(example_index))

    def test_reindex_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(ValueError, match="Length mismatch"):
            s.reindex([*list(example_index), "more_indexes"])

    def test_index_setter_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(ValueError, match="Length mismatch"):
            s.index = [*list(example_index), "more_indexes"]

    def test_shape(self):
        s = lt.Series(example_dict)
        assert s.shape == (len(example_dict),)


class TestSeriesAccessors:
    def test_getitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        assert s[key] == ps[key]
        key = "b"
        assert s[key] == ps[key]
        key = "c"
        assert s[key] == ps[key]

    def test_getitem_collection(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = ["a", "c"]
        assert_series_equal_pandas(s[indexes], ps[indexes])

    def test_getitem_slice(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2)
        assert_series_equal_pandas(s[indexes], ps[indexes])

    def test_getitem_slice_notation(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_series_equal_pandas(s[0:1], ps[0:1])

    def test_getitem_slice_too_many_indexers(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_exception(lambda: ps[0, 1], lambda: s[0, 1], KeyError, match=".*")

    def test_getitem_mask(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        val = 2
        mask_s = s > val
        mask_ps = ps > val
        assert_series_equal_pandas(s[mask_s], ps[mask_ps])

    def test_getitem_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = lt.Series(example_index[:2])
        assert_series_equal_pandas(s[indexes], ps[indexes])

    def test_loc_getitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        assert s.loc[key] == ps.loc[key]
        key = "b"
        assert s.loc[key] == ps.loc[key]
        key = "c"
        assert s.loc[key] == ps.loc[key]

    def test_loc_setitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        value = 4
        s.loc[key] = value
        ps.loc[key] = value
        assert_series_equal_pandas(s, ps)

    def test_loc_getitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a", "b"]
        assert_series_equal_pandas(s.loc[keys], ps.loc[keys])

    def test_loc_setitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a", "b"]
        value = 4
        s.loc[keys] = value
        ps.loc[keys] = value
        assert_series_equal_pandas(s, ps)

    def test_loc_setitem_list_and_collection_value(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a", "b"]
        value = [40, 50]
        s.loc[keys] = value
        ps.loc[keys] = value
        assert_series_equal_pandas(s, ps)

    def test_loc_getitem_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        mask = ["a", "c"]
        mask_s = lt.Series(mask)
        mask_ps = pd.Series(mask)
        assert_series_equal_pandas(s.loc[mask_s], ps.loc[mask_ps])

    def test_loc_setitem_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        index = ["a", "c"]
        value = [0, 1000]
        s_set = lt.Series(value, index=index)
        ps_set = pd.Series(value, index=index)
        s.loc[index] = s_set
        ps.loc[index] = ps_set
        assert_series_equal_pandas(s, ps)

    def test_loc_setitem_series_error(self):
        s = lt.Series(example_dict)
        s_set = lt.Series([0, 1000], index=["a", "c"])
        with pytest.raises(ValueError, match="cannot set using a Series with different index"):
            s.loc["a"] = s_set

    def test_loc_getitem_mask_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        mask = [col == "a" for col in example_dict]
        mask_s = lt.Series(mask, index=example_dict.keys())
        mask_ps = pd.Series(mask, index=example_dict.keys())
        assert_series_equal_pandas(s.loc[mask_s], ps.loc[mask_ps])

    def test_loc_setitem_mapping(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a"]
        value = {"a": 40}
        s.loc[keys] = value
        ps.loc[keys] = value
        assert_series_equal_pandas(s, ps)

    def test_loc_setitem_mapping_error(self):
        s = lt.Series(example_dict)
        keys = ["a"]
        value = {"a": 40, "c": 80}
        with pytest.raises(ValueError, match="cannot set using a Mapping with different keys"):
            s.loc[keys] = value

    def test_loc_get_not_hashable_key(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.loc[{1, 2, 3}]

    def test_loc_set_not_hashable_key(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.loc[{1, 2, 3}] = "no!"

    def test_iloc_getitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        index = 0
        assert s.iloc[index] == ps.iloc[index]
        index = 1
        assert s.iloc[index] == ps.iloc[index]
        index = 2
        assert s.iloc[index] == ps.iloc[index]

    def test_iloc_setitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        index = 0
        value = 4
        s.iloc[index] = value
        ps.iloc[index] = value
        assert_series_equal_pandas(s, ps)

    def test_iloc_getitem_slice(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2, 1)
        assert_series_equal_pandas(s.iloc[indexes], ps.iloc[indexes])

    def test_iloc_setitem_slice(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2, 1)
        value = 4
        s.iloc[indexes] = value
        ps.iloc[indexes] = value
        assert_series_equal_pandas(s, ps)

    def test_iloc_getitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = [0, 1]
        assert_series_equal_pandas(s.iloc[indexes], ps.iloc[indexes])

    def test_iloc_setitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = [0, 1]
        value = 4
        s.iloc[indexes] = value
        ps.iloc[indexes] = value
        assert_series_equal_pandas(s, ps)

    def test_iloc_setitem_list_and_collection_value(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = [0, 1]
        value = [31, 32]
        s.iloc[indexes] = value
        ps.iloc[indexes] = value
        assert_series_equal_pandas(s, ps)

    def test_iloc_getitem_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        mask = [0, 2]
        mask_s = lt.Series(mask)
        mask_ps = pd.Series(mask)
        assert_series_equal_pandas(s.iloc[mask_s], ps.iloc[mask_ps])

    def test_iloc_getitem_mask_series(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        mask = [col == "a" for col in example_dict]
        mask_s = lt.Series(mask, index=example_dict.keys())
        mask_ps = pd.Series(mask, index=example_dict.keys())
        assert_exception(lambda: ps.iloc[mask_ps], lambda: s.iloc[mask_s], ValueError)

    def test_iloc_getitem_mask(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        mask = [col == "a" for col in example_dict]
        assert_series_equal_pandas(s.iloc[mask], ps.iloc[mask])

    def test_iloc_get_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.iloc[{1, 2, 3}]

    def test_loc_set_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.loc[{1, 2, 3}] = "no!"

    def test_head(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        n = 2
        assert_series_equal_pandas(s.head(n), ps.head(n))

    def test_tail(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        n = 2
        assert_series_equal_pandas(s.tail(n), ps.tail(n))

    def test_find(self):
        s = lt.Series(example_dict)
        key = "a"
        assert s.find(example_dict[key]) == key

    def test_find_not_found(self):
        s = lt.Series(example_dict)
        assert s.find("value not found") is None

    def test_ifind(self):
        s = lt.Series(example_dict)
        index = 0
        key = list(example_index)[index]
        assert s.ifind(example_dict[key]) == index

    def test_ifind_not_found(self):
        s = lt.Series(example_dict)
        assert s.ifind("value not found") is None


class TestSeriesMapReduce:
    def test_map(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_series_equal_pandas(s.map(lambda x: x**2), ps.map(lambda x: x**2))

    def test_reduce(self):
        s = lt.Series(example_dict)
        assert s.reduce(lambda acc, cur: acc + cur[0], "") == "".join(example_index)
        s = lt.Series()
        assert s.reduce(lambda *_: 0, 0) == 0

    @pytest.mark.parametrize(
        "func",
        [
            "max",
            "min",
            "sum",
            "all",
            "any",
            "argmax",
            "argmin",
            "idxmax",
            "idxmin",
        ],
    )
    def test_aggregations(self, func):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert getattr(s, func)() == getattr(ps, func)()

    def test_all(self):
        s = lt.Series([0, 1, 2])
        ps = pd.Series([0, 1, 2])
        assert s.all() == ps.all()

    def test_any(self):
        s = lt.Series([0, 1, 2])
        ps = pd.Series([0, 1, 2])
        assert s.any() == ps.any()
        s = lt.Series([0])
        ps = pd.Series([0])
        assert s.any() == ps.any()

    def test_astype(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_series_equal_pandas(s.astype(str), ps.astype(str))

    def test_abs(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert_series_equal_pandas(s.abs(), ps.abs())

    @pytest.mark.parametrize(
        "func",
        [
            "argmax",
            "argmin",
            "idxmax",
            "idxmin",
        ],
    )
    def test_arg_idx_errors(self, func):
        s = lt.Series()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(s, func)()


class TestSeriesStatistics:
    @pytest.mark.parametrize(
        "func",
        [
            "mean",
            "median",
            "std",
            "var",
        ],
    )
    def test_statistics(self, func):
        s = lt.Series(example_stats)
        ps = pd.Series(example_stats)
        assert getattr(s, func)() == getattr(ps, func)()

    def test_statistics_mode(self):
        s = lt.Series(example_stats)
        ps = pd.Series(example_stats)
        assert s.mode() == ps.mode().iloc[0]

    def test_statistics_quantiles(self):
        s = lt.Series(example_stats)
        assert s.quantiles() == statistics.quantiles(example_stats)


class TestSeriesExports:
    def test_to_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert s.to_list() == ps.to_list()

    def test_to_dict(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert s.to_dict() == ps.to_dict()


class TestSeriesComparisons:
    def test_lt_ge(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([1, 2])
        assert_series_equal_pandas(sa < sb, psa < psb)
        assert_series_equal_pandas(sa >= sb, psa >= psb)

    def test_le_gt(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 2])
        assert_series_equal_pandas(sa > sb, psa > psb)
        assert_series_equal_pandas(sa <= sb, psa <= psb)

    def test_eq(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 1])
        assert_series_equal_pandas(sa == sb, psa == psb)
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 2])
        assert_series_equal_pandas(sa == sb, psa == psb)

    def test_ne(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([1, 2])
        assert_series_equal_pandas(sa != sb, psa != psb)
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 1])
        assert_series_equal_pandas(sa != sb, psa != psb)


class TestSeriesOperators:
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
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)

        # Series
        assert_series_equal_pandas(getattr(sa, op)(sb), getattr(psa, op)(psb))
        # Scalar
        assert_series_equal_pandas(getattr(sa, op)(example_scalar), getattr(psa, op)(example_scalar))
        # Collection
        assert_series_equal_pandas(getattr(sa, op)(example_values), getattr(psa, op)(example_values))

    def test_matmul(self):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)

        # Series
        assert ((sa @ sb) == int(psa @ psb)) is True
        # Collection
        assert ((sa @ example_values) == int(psa @ example_values)) is True
        # Right hand operator
        assert ((example_values @ sa) == int(example_values @ psa)) is True

    @pytest.mark.parametrize(
        "op",
        [
            "__and__",
            "__xor__",
            "__or__",
            "__rand__",
            "__rxor__",
            "__ror__",
        ],
    )
    def test_bop(self, op):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)
        # Series
        assert_series_equal_pandas(getattr(sa, op)(sb), getattr(psa, op)(psb))
        # Scalar
        sa = lt.Series(example_dict_a)
        assert_series_equal_pandas(getattr(sa, op)(example_scalar), getattr(psa, op)(example_scalar))
        # Collection
        sa = lt.Series(example_dict_a)
        assert_series_equal_pandas(getattr(sa, op)(example_values), getattr(psa, op)(np.array(example_values)))
        # Pandads is deprecating logical ops for dtype-less sequqences (eg: list, tuple)

    @pytest.mark.parametrize(
        "op",
        [
            "__divmod__",
            "__lshift__",
            "__rshift__",
            "__rdivmod__",
            "__rlshift__",
            "__rrshift__",
        ],
    )
    def test_rop(self, op):
        # Series
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        assert getattr(sa, op)(sb) == lt.Series(
            {k: getattr(v, op)(example_dict_b[k]) for k, v in example_dict_a.items()}
        )

        # Scalar
        sa = lt.Series(example_dict_a)
        assert getattr(sa, op)(example_scalar) == lt.Series(
            {k: getattr(v, op)(example_scalar) for k, v in example_dict_a.items()}
        )

        # Collection
        sa = lt.Series(example_dict_a)
        assert getattr(sa, op)(example_values) == lt.Series(
            {k: getattr(v, op)(example_values[i]) for i, (k, v) in enumerate(example_dict_a.items())}
        )

    @pytest.mark.parametrize(
        "iop",
        [
            "__iadd__",
            "__isub__",
            "__imul__",
            "__itruediv__",
            "__ifloordiv__",
            "__imod__",
            "__ipow__",
            "__iand__",
            "__ixor__",
            "__ior__",
        ],
    )
    def test_op_inplace(self, iop):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)
        getattr(sa, iop)(sb)
        getattr(psa, iop)(psb)
        assert_series_equal_pandas(sa, psa)

    @pytest.mark.parametrize(
        ("iop", "op"),
        [
            ("__ilshift__", "__lshift__"),
            ("__irshift__", "__rshift__"),
        ],
    )
    def test_op_inplace_shift(self, iop, op):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        getattr(sa, iop)(sb)
        assert sa == {k: getattr(v, op)(example_dict_b[k]) for k, v in example_dict_a.items()}

    def test_iop_matmul(self):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        sa @= sb
        assert sa == np.dot(list(example_dict_a.values()), list(example_dict_b.values()))

    def test_different_length_op_error(self):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)[: len(example_dict_b) - 2]
        with pytest.raises(ValueError, match="Cannot operate"):
            sa + sb

    def test_iop_empty(self):
        df = lt.Series()
        df += 10
        assert len(df) == 0


class TestSeriesUnaryOperators:
    def test_neg(self):
        s = lt.Series(example_unary)
        ps = pd.Series(example_unary)
        assert_series_equal_pandas(-s, -ps)

    def test_pos(self):
        s = lt.Series(example_unary)
        ps = pd.Series(example_unary)
        assert_series_equal_pandas(+s, +ps)

    def test_abs(self):
        s = lt.Series(example_unary)
        ps = pd.Series(example_unary)
        assert_series_equal_pandas(abs(s), abs(ps))

    def test_invert(self):
        s = lt.Series(example_unary)
        ps = pd.Series(example_unary)
        assert_series_equal_pandas(~s, ~ps)
