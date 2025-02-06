# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import statistics
from collections import UserDict
from collections.abc import Callable, Collection, Generator, Iterator, Mapping, Sequence
from functools import reduce
from typing import Any, Generic, Literal, Self, TypeAlias, TypeGuard, TypeVar, Union, assert_never, overload

###########################################################################
# Typing
###########################################################################
AxisRows = 0
AxisCols = 1
Axis: TypeAlias = Literal[0, 1]
AxisOrNone: TypeAlias = Axis | None
Scalar: TypeAlias = int | float | complex | str | bool
Index: TypeAlias = Sequence[Scalar]
ArrayLike = Collection
LocIndexes: TypeAlias = Union[Scalar, list[Scalar], slice, "Series"]
IlocIndexes: TypeAlias = Union[int, list[int], slice, "Series"]
LocDataFrameReturn: TypeAlias = Union["DataFrame", "Series", Scalar]
LocSeriesReturn: TypeAlias = Union["Series", Scalar]
T = TypeVar("T", "Series", "DataFrame")


def _is_array_like(value: Any) -> TypeGuard[ArrayLike]:
    return isinstance(value, Collection) and not isinstance(value, (str, DataFrame)) and not _is_scalar(value)


def _is_scalar(value: Any) -> TypeGuard[Scalar]:
    return isinstance(value, (int, float, complex, str, bool))


###########################################################################
# Indexers
###########################################################################
class BaseScalar(Generic[T]):
    data: T

    def __init__(self, data: T):
        self.data = data

    def _is_boolean_mask(self, s: Series | Sequence[object]) -> bool:
        match s:
            case Series():
                return self.data.index == s.index and s.map(lambda v: isinstance(v, bool)).all()
            case Sequence():
                return all(isinstance(b, bool) for b in s)
            case _:  # no cov
                assert_never(s)

    def _index_to_labels(self, index: IlocIndexes) -> Scalar | list[Scalar]:
        match index:
            case int():
                return self.data.index[index]
            case slice():
                return list(self.data.index[index])
            case list():
                if self._is_boolean_mask(index):
                    return [self.data.index[i] for i, v in enumerate(index) if v is True]
                return [self.data.index[i] for i in index]
            case Series():
                if self._is_boolean_mask(index):
                    msg = "iLocation based boolean indexing cannot use an indexable as a mask"
                    raise ValueError(msg)
                return [self.data.index[i] for i in index.values]
        msg = f"Cannot index with unhashable: {index=}"
        raise TypeError(msg)


class LocSeriesIndexer(BaseScalar["Series"]):
    def __getitem__(self, label: LocIndexes) -> Scalar | Series:
        match label:
            case list():
                return Series({k: self.data[k] for k in label}, name=self.data.name)
            case slice():
                return Series({k: self.data[k] for k in self.data.index[label]}, name=self.data.name)
            case Series():
                if self._is_boolean_mask(label):
                    return Series({k: v for k, v in self.data.items() if label[k]}, name=self.data.name)
                return Series({k: self.data[k] for k in label.values}, name=self.data.name)
            case _ if _is_scalar(label):
                return self.data[label]
        msg = f"Cannot index with unhashable: {label=}"
        raise TypeError(msg)

    def __setitem__(self, label: LocIndexes, value: Scalar | ArrayLike | Mapping | Series):
        labels: list[Scalar]
        match label:
            case list():
                labels = label
            case Series():
                labels = list(label.values)
            case slice():
                labels = list(self.data.index[label])
            case _ if _is_scalar(label):
                labels = [label]
            case _:
                msg = f"Cannot index with unhashable: {label=}"
                raise TypeError(msg)

        values: list[Any]
        match value:
            case Series():
                values = list(value.values)
                if set(labels) != set(value.index):
                    msg = "cannot set using a Series with different index"
                    raise ValueError(msg)
            case Mapping():
                if set(labels) != set(value.keys()):
                    msg = "cannot set using a Mapping with different keys"
                    raise ValueError(msg)
                values = [value[b] for b in labels]
            case v if _is_array_like(v):
                values = list(value)
            case v if _is_scalar(v):
                values = [value for _ in range(len(labels))]
            case _ as unreachable:  # no cov
                assert_never(unreachable)  # type: ignore # @TODO: How to exhaust this check?

        for k, v in zip(labels, values):
            self.data[k] = v


class IlocSeriesIndexer(BaseScalar["Series"]):
    @overload
    def __getitem__(self, index: int) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, index: list[int] | slice | Series) -> Series: ...  # no cov
    def __getitem__(self, index: IlocIndexes) -> Scalar | Series:
        return self.data.loc.__getitem__(self._index_to_labels(index))

    def __setitem__(self, index: IlocIndexes, value: Scalar | ArrayLike | Mapping | Series):
        self.data.loc.__setitem__(self._index_to_labels(index), value)


class LocDataFrameIndexer(BaseScalar["DataFrame"]):
    def _select_columns(self, label: LocIndexes) -> DataFrame | Series:
        match label:
            case list():
                return DataFrame({k: self.data[k] for k in label}, columns=label)
            case slice():
                columns = self.data.columns[label]
                return DataFrame({k: self.data[k] for k in columns}, columns=columns)
            case Series():
                columns = label.values
                return DataFrame({k: self.data[k] for k in columns}, columns=columns)
            case _ if _is_scalar(label):
                return self.data[label]
        msg = f"Cannot index with unhashable: {label=}"
        raise TypeError(msg)

    @overload
    def _columns_to_labels(self, index: int) -> Scalar: ...  # no cov
    @overload
    def _columns_to_labels(self, index: slice | list[int] | Series) -> list[Scalar]: ...  # no cov
    def _columns_to_labels(self, index: IlocIndexes) -> Scalar | list[Scalar]:
        match index:
            case int():
                return self.data.columns[index]
            case slice():
                return list(self.data.columns[index])
            case list():
                return [self.data.columns[i] for i in index]
            case Series():
                return [self.data.columns[i] for i in index]
        msg = f"Cannot index with unhashable: {index=}"
        raise TypeError(msg)

    @overload
    def __getitem__(self, label: tuple[Scalar, Scalar]) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, label: Scalar | tuple[Scalar, Any] | tuple[Any, Scalar]) -> Series: ...  # no cov
    @overload
    def __getitem__(self, label: Any) -> DataFrame: ...  # no cov
    def __getitem__(self, label: LocIndexes | tuple[LocIndexes, LocIndexes]) -> LocDataFrameReturn:
        match label:
            case (row_labels, col_labels) if isinstance(label, tuple):
                data = self._select_columns(col_labels)
                return data.loc[row_labels]
            case list() | slice() | Series():
                return DataFrame({col: self.data[col].loc[label] for col in self.data.columns})
            case _ if _is_scalar(label):
                return Series(
                    {col: self.data[col][label] for col in self.data.columns}, index=self.data.columns, name=label
                )
            case _:
                msg = f"Cannot index with unhashable: {label=}"
                raise TypeError(msg)

    def __setitem__(self, label: LocIndexes | tuple[LocIndexes, LocIndexes], value: Scalar | ArrayLike | Mapping):
        col_labels: LocIndexes
        row_labels: LocIndexes
        match label:
            case (row_labels, col_labels) if isinstance(label, tuple):
                match col_labels:
                    case slice():
                        col_labels = self._columns_to_labels(col_labels)
                    case Series():
                        col_labels = list(col_labels.values)
                    case list():
                        pass
                    case c if _is_scalar(c):
                        col_labels = [c]
                    case _:
                        msg = f"Cannot index with unhashable: {label=}"
                        raise TypeError(msg)
            case list() | slice() | Series():
                row_labels = label
                col_labels = list(self.data.columns)
            case l if _is_scalar(l):
                row_labels = l
                col_labels = list(self.data.columns)
            case _:
                msg = f"Cannot index with unhashable: {label=}"
                raise TypeError(msg)

        match value:
            case v if _is_scalar(v):
                for col in col_labels:
                    self.data.data[col].loc[row_labels] = value
            case Mapping():
                if set(value.keys()) != set(col_labels):
                    msg = "cannot set using a Mapping with different keys"
                    raise ValueError(msg)
                for col, val in value.items():
                    self.data.data[col].loc[row_labels] = val
            case v if _is_array_like(v):
                if len(value) != len(col_labels):
                    msg = "cannot set using a list-like indexer with a different length than the value"
                    raise ValueError(msg)
                for col, val in zip(col_labels, value):
                    self.data.data[col].loc[row_labels] = val
            case _:  # no cov
                msg = f"Cannot set with: {value=}"
                raise TypeError(msg)


class IlocDataFrameIndexer(BaseScalar["DataFrame"]):
    @overload
    def __getitem__(self, index: tuple[int, int]) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, index: int | tuple[int, Any] | tuple[Any, int]) -> Series: ...  # no cov
    @overload
    def __getitem__(self, index: Any) -> DataFrame: ...  # no cov
    def __getitem__(self, index: IlocIndexes | tuple[IlocIndexes, IlocIndexes]) -> LocDataFrameReturn:
        match index:
            case (row_indexes, col_indexes) if isinstance(index, tuple):
                return self.data.loc.__getitem__(
                    (self._index_to_labels(row_indexes), self.data.loc._columns_to_labels(col_indexes))  # noqa: SLF001
                )
            case _:
                return self.data.loc.__getitem__(self._index_to_labels(index))  # type: ignore

    def __setitem__(self, index: IlocIndexes, value: Scalar | ArrayLike | Mapping):
        match index:
            case (row_indexes, col_indexes) if isinstance(index, tuple):
                return self.data.loc.__setitem__(
                    (self._index_to_labels(row_indexes), self.data.loc._columns_to_labels(col_indexes)),  # noqa: SLF001
                    value,
                )
            case _:
                return self.data.loc.__setitem__(self._index_to_labels(index), value)


###########################################################################
# Series
###########################################################################
class Series(UserDict):
    """
    Series class representing a one-dimensional labeled array with capabilities for data analysis.

    Attributes:
        name (Scalar): Name of the Series.
    """

    name: Scalar | None
    loc: LocSeriesIndexer
    iloc: IlocSeriesIndexer
    __slots__ = ("name", "loc", "iloc")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self, data: Mapping | ArrayLike | Scalar | None = None, index: Index | None = None, name: Scalar | None = None
    ):
        """
        Initializes a Series object.

        Args:
            data (Mapping | ArrayLike | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (Index, optional): Index for the Series. Defaults to None.
            name (Scalar, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if data is None:
            super().__init__()
        elif isinstance(data, Mapping):
            if index is not None:
                data = {k: v for k, v in data.items() if k in index}
            super().__init__(data)
        elif isinstance(data, Scalar):
            super().__init__({0: data})
        elif isinstance(data, ArrayLike):
            if index is None:
                index = range(len(data))
            elif len(data) != len(list(index)):
                msg = f"Length of values ({len(data)}) does not match length of index ({len(index)})"
                raise ValueError(msg)
            super().__init__(dict(zip(index, data)))
        else:
            msg = f"Unexpected data type: {type(data)=}"
            raise ValueError(msg)
        self.name = name
        self._set_indexers()

    def _set_indexers(self):
        self.loc = LocSeriesIndexer(self)
        self.iloc = IlocSeriesIndexer(self)

    def __repr__(self) -> str:
        if len(self) == 0:
            if self.name is None:
                return "Empty Series"
            return f'Empty Series(name="{self.name}")'
        columns = list(zip(*self.items()))
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret) + f"\nname: {self.name}\n"

    def copy(self, *, deep: bool = True):
        """
        Creates a copy of the Series.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            Series: A copy of the Series.
        """
        clone = copy.deepcopy(self) if deep else copy.copy(self)
        clone.name = self.name
        clone._set_indexers()  # noqa: SLF001
        return clone

    def rename(self, name: Scalar) -> Series:
        """
        Renames the Series.

        Args:
            name (Scalar): The new name for the Series.

        Returns:
            Series: A new Series with the updated name (a copy).
        """
        clone = self.copy(deep=True)
        clone.name = name
        return clone

    @property
    def index(self) -> Index:
        """
        Returns the index of the Series.

        Returns:
            Index: The index of the Series.
        """
        return list(self.keys())

    @index.setter
    def index(self, index: Index):
        """
        Sets the index of the Series.

        Args:
            value (Index): The new index for the Series.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        self.data = dict(zip(index, self.values))

    def reindex(self, index: Index) -> Series:
        """
        Sets the index of the Series.

        Args:
            value (Index): The new index for the Series.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        clone = self.copy(deep=True)
        clone.data = dict(zip(index, self.values))
        return clone

    @property
    def values(self) -> list[Any]:  # type: ignore
        """
        Return a list representation of the Series.

        Returns:
            list: The values of the Series.
        """
        return list(self.data.values())

    @property
    def shape(self) -> tuple[int,]:
        return (len(self.index),)

    ###########################################################################
    # Accessors
    ###########################################################################
    def __getitem__(self, name: Scalar | list[Scalar] | slice | Series) -> Any | Series:
        """
        Retrieves an item or slice from the Series.

        Args:
            name (Scalar | list[Scalar] | slice): The key, list of keys, or slice to retrieve.

        Returns:
            Any: The value(s) associated with the given key(s) or slice.
            Series: A new Series if a list or slice is provided.
        """
        if isinstance(name, (list, Series)):
            return self.loc[name]
        if isinstance(name, slice):
            return self.iloc[name]
        return super().__getitem__(name)

    def head(self, n: int = 5) -> Series:
        """
        Returns the first n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the first n rows.
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> Series:
        """
        Returns the last n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the last n rows.
        """
        return self.iloc[-n:]

    def ifind(self, val: Any) -> int | None:
        """
        Finds the first integer position (index) of a given value in the Series.

        Args:
            val (Any): The value to search for.

        Returns:
            int | None: The integer position (index) of the first occurrence of the value,
                        or None if the value is not found.
        """
        for i, v in enumerate(self.values):
            if v == val:
                return i
        return None

    def find(self, val: Any) -> Scalar | None:
        """
        Finds the first label (key) associated with a given value in the Series.

        Args:
            val (Any): The value to search for.

        Returns:
            Scalar | None: The label (key) of the first occurrence of the value,
                             or None if the value is not found.
        """
        for k, v in self.items():
            if v == val:
                return k
        return None

    ###########################################################################
    # Auxiliary Functions
    ###########################################################################
    def _other_as_series(self, other: Series | Scalar | ArrayLike) -> Series:
        """Converts other to a Series if it is not already. Used for operations."""
        if isinstance(other, Series):
            return other
        if isinstance(other, Scalar):
            return Series([other] * len(self), index=self.index, name=self.name)
        if isinstance(other, ArrayLike):
            return Series(other, index=self.index, name=self.name)
        return NotImplemented  # no cov

    def _other_as_series_matching(self, other: Series | ArrayLike | Scalar) -> Series:
        """Converts and matches index of other to self. Used for operations."""
        other = self._other_as_series(other)
        if self.index != other.index:
            msg = "Cannot operate in Series with different index"
            raise ValueError(msg)
        return other

    ###########################################################################
    # Map/Reduce
    ###########################################################################
    def map(self, func: Callable) -> Series:
        """
        Applies a function to each value in the Series.

        Args:
            func (Callable): The function to apply.

        Returns:
            Series: A new Series with the results of the function applied.
        """
        return Series({k: func(v) for k, v in self.items()})

    def reduce(self, func: Callable, initial: Any):
        """
        Reduces the Series using a function.

        Args:
            func (Callable): The function to apply for reduction.
            initial (Any): The initial value for the reduction.

        Returns:
            Any: The reduced value.
        """
        if len(self) > 0:
            return reduce(func, self.items(), initial)
        return initial

    def agg(self, func: Callable) -> Any:
        """
        Applies an aggregation function to the Series' values.

        This method applies a given function to all the values in the Series.
        It is intended for aggregation functions that operate on a collection
        of values and return a single result.

        Args:
            func (Callable): The aggregation function to apply. This function
                should accept an iterable (like a list or NumPy array) and
                return a single value.

        Returns:
            Any: The result of applying the aggregation function to the Series' values.
        """
        return func(self.values)

    def astype(self, new_type: type) -> Series:
        """
        Casts the Series to a new type.

        Args:
            new_type (type): The type to cast to.

        Returns:
            Series: A new Series with the values cast to the new type.
        """
        return self.map(new_type)

    def abs(self) -> Series:
        """
        Returns the absolute values for Series

        Returns:
            Series: Absolute values Series
        """
        return self.map(abs)

    def dot(self, other: Series | ArrayLike) -> Scalar:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Series by the scalar and returns the sum.

        Args:
            other (Series | ArrayLike | Scalar)

        Returns:
            Scalar: The dot product of the Series.
        """
        other = self._other_as_series_matching(other)
        return sum(other[key] * value for key, value in self.items())

    def max(self) -> Scalar:
        """
        Returns the maximum value in the Series.

        Returns:
            Any: The maximum value.
        """
        return self.agg(max)

    def min(self) -> Scalar:
        """
        Returns the minimum value in the Series.

        Returns:
            Any: The minimum value.
        """
        return self.agg(min)

    def sum(self) -> Scalar:
        """
        Returns the sum of the values in the Series.

        Returns:
            Any: The sum of the values.
        """
        return self.agg(sum)

    def all(self) -> bool:
        """
        Returns True if all values in the Series are True.

        Returns:
            bool: True if all values are True, False otherwise.
        """
        return self.agg(all)

    def any(self) -> bool:
        """
        Returns True if any value in the Series is True.

        Returns:
            bool: True if any value is True, False otherwise.
        """
        return self.agg(any)

    def argmax(self) -> int:
        """
        Returns the index of the maximum value.

        Returns:
            int: The index of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmax of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.max())  # type: ignore

    def argmin(self) -> int:
        """
        Returns the index of the minimum value.

        Returns:
            int: The index of the minimum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmin of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.min())  # type: ignore

    def idxmax(self) -> Scalar | None:
        """
        Returns the label of the maximum value.

        Returns:
            Scalar: The label of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get ixmax of an empty sequence"
            raise ValueError(msg)
        return self.find(self.max())

    def idxmin(self) -> Scalar | None:
        """
        Returns the label of the minimum value.

        Returns:
            Scalar: The label of the minimum value.
        """
        if len(self) == 0:
            msg = "Attempt to get idxmin of an empty sequence"
            raise ValueError(msg)
        return self.find(self.min())

    ###########################################################################
    # Statistics
    ###########################################################################
    def mean(self) -> Scalar:
        """
        Computes the mean of the Series.

        Returns:
            float: Series mean
        """
        return self.agg(statistics.mean)

    def median(self) -> Scalar:
        """
        Return the median (middle value) of numeric data, using the common “mean of middle two” method.
        If data is empty, StatisticsError is raised. data can be a sequence or iterable.

        Returns:
            float | int: Series median
        """
        return self.agg(statistics.median)

    def mode(self) -> Scalar:
        """
        Return the single most common data point from discrete or nominal data. The mode (when it exists)
        is the most typical value and serves as a measure of central location.

        Returns:
            Any: Series mode
        """
        return self.agg(statistics.mode)

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive") -> ArrayLike[float]:
        """
        Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
        cut points separating the intervals.

        Returns:
            list[float]: List containing quantiles
        """
        return self.agg(lambda values: statistics.quantiles(values, n=n, method=method))

    def std(self, xbar=None) -> Scalar:
        """
        Return the sample standard deviation (the square root of the sample variance).
        See variance() for arguments and other details.

        Returns:
            float: Series standard deviation
        """
        return self.agg(lambda values: statistics.stdev(values, xbar=xbar))

    def var(self, xbar=None) -> Scalar:
        """
        Return the sample variance of data, an iterable of at least two real-valued numbers.
        Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series variance
        """
        return self.agg(lambda values: statistics.variance(values, xbar=xbar))

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[Any]:
        """
        Converts the Series to a list.

        Returns:
            list[Any]: A list of the Series values.
        """
        return self.values

    def to_dict(self) -> dict[Scalar, Any]:
        """
        Converts the Series to a dictionary.

        Returns:
            dict[Scalar, Any]: A dictionary representation of the Series.
        """
        return dict(self)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def __lt__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise less than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v < other[k] for k, v in self.items()}, name=self.name)

    def __le__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise less than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v <= other[k] for k, v in self.items()}, name=self.name)

    def __eq__(self, other: Series | ArrayLike | Scalar) -> Series:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v == other[k] for k, v in self.items()}, name=self.name)

    def __ne__(self, other: Series | ArrayLike | Scalar) -> Series:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v != other[k] for k, v in self.items()}, name=self.name)

    def __gt__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise greater than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v > other[k] for k, v in self.items()}, name=self.name)

    def __ge__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v >= other[k] for k, v in self.items()}, name=self.name)

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise addition.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v + other[k] for k, v in self.items()}, name=self.name)

    def __sub__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise subtraction.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v - other[k] for k, v in self.items()}, name=self.name)

    def __mul__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise multiplication.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v * other[k] for k, v in self.items()}, name=self.name)

    def __matmul__(self, other: Series | ArrayLike) -> Scalar:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.

        Args:
            other (Series | ArrayLike)

        Returns:
            Scalar: The dot product of the Series.
        """
        return self.dot(other)

    def __truediv__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise division.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v / other[k] for k, v in self.items()}, name=self.name)

    def __floordiv__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise floor division.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v // other[k] for k, v in self.items()}, name=self.name)

    def __mod__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise modulo.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v % other[k] for k, v in self.items()}, name=self.name)

    def __divmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise divmod.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: divmod(v, other[k]) for k, v in self.items()}, name=self.name)

    def __pow__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise exponentiation.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: pow(v, other[k]) for k, v in self.items()}, name=self.name)

    def __lshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise left bit shift.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v << other[k] for k, v in self.items()}, name=self.name)

    def __rshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise right bit shift.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v >> other[k] for k, v in self.items()}, name=self.name)

    def __and__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise AND.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v & other[k] for k, v in self.items()}, name=self.name)

    def __xor__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise XOR.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v ^ other[k] for k, v in self.items()}, name=self.name)

    def __or__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise OR.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v | other[k] for k, v in self.items()}, name=self.name)

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    def __radd__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other + self

    def __rsub__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other - self

    def __rmul__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other * self

    def __rmatmul__(self, other: Series | ArrayLike) -> Scalar:
        return self.dot(other)

    def __rtruediv__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other / self

    def __rfloordiv__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other // self

    def __rmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other % self

    def __rdivmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return divmod(other, self)

    def __rpow__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return pow(other, self)

    def __rlshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other << self

    def __rrshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other >> self

    def __rand__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other & self

    def __rxor__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other ^ self

    def __ror__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other | self

    ###########################################################################
    # In-place Operators
    ###########################################################################
    def __iadd__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] += other[k]
        return self

    def __isub__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] -= other[k]
        return self

    def __imul__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] *= other[k]
        return self

    def __imatmul__(self, other: Series | ArrayLike) -> Scalar:  # noqa: PYI034
        return self.dot(other)

    def __itruediv__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] /= other[k]
        return self

    def __ifloordiv__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] //= other[k]
        return self

    def __imod__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] %= other[k]
        return self

    def __ipow__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] **= other[k]
        return self

    def __ilshift__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] <<= other[k]
        return self

    def __irshift__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] >>= other[k]
        return self

    def __iand__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] &= other[k]
        return self

    def __ixor__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] ^= other[k]
        return self

    def __ior__(self, other: Series | ArrayLike | Scalar) -> Self:  # type: ignore
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] |= other[k]
        return self

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> Series:
        return Series({k: -v for k, v in self.items()})

    def __pos__(self) -> Series:
        return Series({k: +v for k, v in self.items()})

    def __abs__(self) -> Series:
        return self.abs()

    def __invert__(self) -> Series:
        return Series({k: ~v for k, v in self.items()})


###########################################################################
# DataFrame
###########################################################################
class DataFrame(UserDict):
    """
    DataFrame class representing a two-dimensional, size-mutable, tabular data structure with
    labeled axes (rows and columns).

    Attributes:
        index (Index): The row labels (index) of the DataFrame. Used for label-based row selection.
        columns (Index): The column labels of the DataFrame. Used for label-based column selection.
    """

    _index: Index
    _columns: Index
    loc: LocDataFrameIndexer
    iloc: IlocDataFrameIndexer
    data: dict[Scalar, Series]
    __slots__ = ("_index", "_columns", "loc", "iloc")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self,
        data: Mapping[Scalar, Series]
        | Mapping[Scalar, ArrayLike[Scalar]]
        | ArrayLike[Series]
        | ArrayLike[Mapping[Scalar, Scalar]]
        | ArrayLike[Scalar]
        | ArrayLike[ArrayLike[Scalar]]
        | Iterator
        | None = None,
        index: Index | None = None,
        columns: Index | None = None,
    ):
        """
        Initializes a DataFrame object.

        Args:
            data (Mapping | ArrayLike | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (Index, optional): Index for the Series. Defaults to None.
            name (Scalar, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if isinstance(data, Iterator):
            data = list(data)
        match data:
            case None:
                self._init_empty(index, columns)
            case Mapping() | ArrayLike() if len(data) == 0:
                self._init_empty(index, columns)
            case Mapping() as m if all(isinstance(v, (Series, ArrayLike)) for v in m.values()):
                self._init_mapping_of_series({col: Series(val) for col, val in data.items()}, index, columns)  # type: ignore
            case ArrayLike() as c if all(isinstance(v, Series) for v in c):
                self._init_collection_of_series(data, index, columns)  # type: ignore
            case ArrayLike() as c if all(isinstance(v, Mapping) for v in c):
                self._init_collection_of_series([Series(row) for row in data], index, columns)  # type: ignore
            case ArrayLike() as c if all(isinstance(v, Scalar) for v in c) and not isinstance(c, str):
                self._init_mapping_of_series({0: Series(data)}, index, columns)
            case ArrayLike() as c if all(isinstance(v, ArrayLike) for v in c) and not isinstance(c, str):
                self._init_collection_of_series([Series(row) for row in data], index, columns)  # type: ignore
            case _:
                msg = "DataFrame constructor not properly called!"
                raise ValueError(msg)
        self._validate_index_and_columns()
        self._set_indexers()

    def _init_empty(self, index: Index | None = None, columns: Index | None = None):
        super().__init__()
        if (index is not None and len(index) > 0) or (columns is not None and len(columns) > 0):
            msg = "Cannot create an empty DataFrame with preset columns and/or indexes"
            raise ValueError(msg)
        self._index = []
        self._columns = []

    def _init_mapping_of_series(
        self, data: Mapping[Scalar, Series], index: Index | None = None, columns: Index | None = None
    ):
        col0 = next(iter(data))
        val0 = data[col0]
        if len(val0) == 0:
            self._init_empty(index, columns)
            return

        self._index = val0.index if index is None else index
        self._columns = list(data.keys()) if columns is None else columns
        if (len(self._index) != len(val0.index)) or (len(self._columns) != len(data)):
            passed = (len(val0.index), len(data))
            implied = (len(self._index), len(self._columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)
        super().__init__(
            {col: s.copy().rename(col).reindex(self._index) for col, s in zip(self._columns, data.values())}
        )

    def _init_collection_of_series(
        self, data: ArrayLike[Series], index: Index | None = None, columns: Index | None = None
    ):
        row0 = next(iter(data))
        src_columns = row0.index
        if len(src_columns) == 0:
            self._init_empty(index, columns)
            return

        if columns is not None:
            self._columns = columns
        else:
            self._columns = src_columns
        if any(d.index != src_columns for d in data):
            all_cols = {item for d in data for item in d.index}
            missing_cols = all_cols - set(self._columns) or "{}"
            extra_cols = set(self._columns) - all_cols or "{}"
            msg = f"Misaligned columns. Expected {self._columns}. Missing: {missing_cols}, Extra: {extra_cols}"
            raise ValueError(msg)

        # @TODO: Deal with Series with names
        self._index = list(range(len(data))) if index is None else index
        if (len(self._index) != len(data)) or (len(self._columns) != len(src_columns)):
            passed = (len(data), len(src_columns))
            implied = (len(self._index), len(self._columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)
        super().__init__(
            {
                dst_col: Series({idx: row[src_col] for idx, row in zip(self._index, data)}, name=dst_col)
                for src_col, dst_col in zip(src_columns, self._columns)
            }
        )

    def _validate_index_and_columns(self):
        for col, s in self.items():
            if s.index != self._index:
                msg = "Somehow the inner indexes and DataFrame indexex don't match. This shouldn't happen!"
                raise ValueError(msg)
            if s.name != col:
                msg = "Somehow the inner columns and DataFrame columns don't match. This shouldn't happen!"
                raise ValueError(msg)

    def _set_indexers(self):
        self.iloc = IlocDataFrameIndexer(self)
        self.loc = LocDataFrameIndexer(self)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.index), len(self.columns))

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self):
        if len(self) == 0:
            return "Empty DataFrame"
        columns = list(zip(["", *self.columns], *[[col, *s.values] for col, s in self.iterrows()]))
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret)

    def copy(self, *, deep: bool = True) -> DataFrame:
        """
        Creates a copy of the DataFrame.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            DataFrame: A copy of the DataFrame.
        """
        clone = copy.deepcopy(self) if deep else copy.copy(self)
        clone._set_indexers()  # noqa: SLF001
        return clone

    @property
    def index(self) -> Index:
        """
        Returns the index of the DataFrame.

        Returns:
            Index: The index of the DataFrame.
        """
        return self._index

    @index.setter
    def index(self, index: Index):
        """
        Sets the index of the DataFrame.

        Args:
            value (Index): The new index for the DataFrame.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        self._index = index
        for s in self.data.values():
            s.index = index

    @property
    def columns(self) -> Index:
        """
        Returns the columns of the DataFrame.

        Returns:
            Index: The columns of the DataFrame.
        """
        return self._columns

    @columns.setter
    def columns(self, columns: Index):
        """
        Sets the columns of the DataFrame.

        Args:
            columns (Index): The new index for the DataFrame.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self.columns) != len(columns):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(columns)} elements"
            raise ValueError(msg)
        self._columns = columns

    @property
    def T(self) -> DataFrame:  # noqa: N802
        data = list(self.data.values())
        return DataFrame(data, index=self.columns[:], columns=self.index[:])

    @property
    def values(self) -> list[list[Any]]:  # type: ignore
        """
        Return a list representation of the DataFrame.

        Returns:
            list: The values of the DataFrame.
        """
        return self.to_list()

    def iterrows(self) -> Generator[tuple[Scalar, Series]]:
        yield from self.T.items()

    ###########################################################################
    # Accessors
    ###########################################################################
    @overload
    def __getitem__(self, index: Scalar) -> Series: ...  # no cov
    @overload
    def __getitem__(self, index: list[Scalar] | slice | Series) -> DataFrame: ...  # no cov
    def __getitem__(self, index: Scalar | list[Scalar] | slice | Series) -> LocDataFrameReturn:
        """
        Retrieves an item or slice from the DataFrame.

        Args:
            index (Scalar | list[Scalar] | slice): The key, list of keys, or slice to retrieve.

        Returns:
            Series | DataFRame: A new Series if a single column is selected or a DataFrame for multiple columns
        """
        if isinstance(index, (list, Series)):
            if self.loc._is_boolean_mask(Series(index)):  # noqa: SLF001
                return self.loc[index]
            return self.loc[:, index]
        if isinstance(index, slice):
            return self.iloc[index]
        return super().__getitem__(index)

    def head(self, n: int = 5) -> DataFrame:
        """
        Returns the first n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            DataFrame: A new DataFrame containing the first n rows.
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> DataFrame:
        """
        Returns the last n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            DataFrame: A new DataFrame containing the last n rows.
        """
        return self.iloc[-n:]

    ###########################################################################
    # Apply/Agg/Map/Reduce
    ###########################################################################
    def apply(self, method: Callable[[Series], Any], axis: Axis = 0) -> Series:
        match axis:
            case int(c) if c == AxisRows:
                return Series({col: method(s) for col, s in self.items()})
            case int(c) if c == AxisCols:
                return self.T.apply(method, axis=0)
            case _:
                msg = f"No axis named {axis} for object type DataFrame"
                raise ValueError(msg)

    def _apply_with_none(self, method: Callable[[Series], Any], axis: AxisOrNone = 0):
        match axis:
            case None:
                return method(self.apply(method, 0))
            case _:
                return self.apply(method, axis)

    def agg(self, method: Callable[[ArrayLike[Any]], Any], axis: Axis = 0) -> Series:
        match axis:
            case int(c) if c == AxisRows:
                return Series({col: method(s.values) for col, s in self.items()})
            case int(c) if c == AxisCols:
                return self.T.agg(method, axis=0)
            case _:
                msg = f"No axis named {axis} for object type DataFrame"
                raise ValueError(msg)

    def _agg_with_none(self, method: Callable[[ArrayLike[Any]], Any], axis: AxisOrNone = 0):
        match axis:
            case None:
                return method([item for sublist in self.to_list() for item in sublist])
            case _:
                return self.agg(method, axis)

    def map(self, func: Callable) -> DataFrame:
        """
        Applies a function to each value in the DataFrame.

        Args:
            func (Callable): The function to apply.

        Returns:
            DataFrame: A new DataFrame with the results of the function applied.
        """
        return DataFrame({col: s.map(func) for col, s in self.items()})

    def astype(self, new_type: type) -> DataFrame:
        """
        Casts the DataFrame to a new type.

        Args:
            new_type (type): The type to cast to.

        Returns:
            DataFrame: A new DataFrame with the values cast to the new type.
        """
        return self.map(new_type)

    # @overload
    # def dot(self, other: DataFrame) -> DataFrame: ...  # no cov
    # @overload
    # def dot(self, other: Series | ArrayLike) -> Series: ...  # no cov
    def dot(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:
        """
        Compute the matrix multiplication between the DataFrame and another DataFrame or Series.

        Parameters:
            other (DataFrame or Series): The other DataFrame or Series to multiply with.

        Returns:
            DataFrame or Series: The result of the matrix multiplication.

        Raises:
            ValueError: If the columns of the DataFrame do not match the index of the other DataFrame/Series.
            TypeError: If the other object is not a DataFrame or Series.
        """
        not_aligned_msg = "matrices are not aligned"
        match other:
            case DataFrame():
                if list(self.columns) != list(other.index):
                    raise ValueError(not_aligned_msg)
                df_data = [[s_a.dot(s_b) for s_b in other.data.values()] for s_a in self.T.data.values()]
                return DataFrame(df_data, index=self.index, columns=other.columns)
            case Series() | ArrayLike():
                if len(self.columns) != len(other):
                    raise ValueError(not_aligned_msg)
                if not isinstance(other, Series):
                    other = Series(other, index=self.columns)
                if list(self.columns) != list(other.index):
                    raise ValueError(not_aligned_msg)
                s_data = [sa.dot(other) for sa in self.T.data.values()]
                return Series(s_data, index=self.index, name=other.name)
            case _:
                msg = "Dot product requires other to be a DataFrame or Series."
                raise TypeError(msg)

    def abs(self) -> DataFrame:
        """
        Returns the absolute values for DataFrame

        Returns:
            DataFrame: Absolute values DataFrame
        """
        return self.map(abs)

    @overload
    def max(self) -> Series: ...  # no cov
    @overload
    def max(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def max(self, axis: None) -> Scalar: ...  # no cov
    def max(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.max(), axis)

    @overload
    def min(self) -> Series: ...  # no cov
    @overload
    def min(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def min(self, axis: None) -> Scalar: ...  # no cov
    def min(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.min(), axis)

    @overload
    def sum(self) -> Series: ...  # no cov
    @overload
    def sum(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def sum(self, axis: None) -> Scalar: ...  # no cov
    def sum(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.sum(), axis)

    @overload
    def all(self) -> Series: ...  # no cov
    @overload
    def all(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def all(self, axis: None) -> bool: ...  # no cov
    def all(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.all(), axis)

    @overload
    def any(self) -> Series: ...  # no cov
    @overload
    def any(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def any(self, axis: None) -> bool: ...  # no cov
    def any(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.any(), axis)

    @overload
    def idxmax(self) -> Series: ...  # no cov
    @overload
    def idxmax(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def idxmax(self, axis: None) -> bool: ...  # no cov
    def idxmax(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.idxmax(), axis)

    @overload
    def idxmin(self) -> Series: ...  # no cov
    @overload
    def idxmin(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def idxmin(self, axis: None) -> bool: ...  # no cov
    def idxmin(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.idxmin(), axis)

    ###########################################################################
    # Statistics
    ###########################################################################
    @overload
    def mean(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def mean(self, axis: None) -> Scalar: ...  # no cov
    def mean(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Computes the mean of the Series.

        Returns:
            float: Series mean
        """
        return self._agg_with_none(statistics.mean, axis=axis)

    @overload
    def median(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def median(self, axis: None) -> Scalar: ...  # no cov
    def median(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the median (middle value) of numeric data, using the common “mean of middle two” method.
        If data is empty, StatisticsError is raised. data can be a sequence or iterable.

        Returns:
            float | int: Series median
        """
        return self._agg_with_none(statistics.median, axis=axis)

    def mode(self, axis: Axis = 0) -> Series:
        """
        Return the single most common data point from discrete or nominal data. The mode (when it exists)
        is the most typical value and serves as a measure of central location.

        Returns:
            Any: Series mode
        """
        # @TOOO: Improve this. Might have to implement NaNs
        return self.agg(statistics.mode, axis=axis)

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive", axis: Axis = 0) -> Series:
        """
        Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
        cut points separating the intervals.

        Returns:
            list[float]: List containing quantiles
        """
        return self.agg(lambda values: statistics.quantiles(values, n=n, method=method), axis=axis)

    @overload
    def std(self, xbar, axis: Axis) -> Series: ...  # no cov
    @overload
    def std(self, xbar, axis: None) -> Scalar: ...  # no cov
    def std(self, xbar=None, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the sample standard deviation (the square root of the sample variance).
        See variance() for arguments and other details.

        Returns:
            float: Series standard deviation
        """
        return self._agg_with_none(lambda values: statistics.stdev(values, xbar=xbar), axis=axis)

    @overload
    def var(self, xbar, axis: Axis) -> Series: ...  # no cov
    @overload
    def var(self, xbar, axis: None) -> Scalar: ...  # no cov
    def var(self, xbar=None, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the sample variance of data, an iterable of at least two real-valued numbers.
        Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series variance
        """
        return self._agg_with_none(lambda values: statistics.variance(values, xbar=xbar), axis=axis)

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[list[Any]]:
        """
        Converts the DataFrame to a list.

        Returns:
            list[list[Any]]: A list of the Series values.
        """
        return list(self.apply(lambda s: s.values, axis=1).data.values())

    @overload
    def to_dict(self) -> dict[Scalar, dict[Scalar, Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["dict"]) -> dict[Scalar, dict[Scalar, Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["list"]) -> dict[Scalar, list[Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["records"]) -> list[dict[Scalar, Any]]: ...  # no cov
    def to_dict(self, orient: Literal["dict", "list", "records"] = "dict"):
        """
        Converts the DataFrame to a dictionary.

        Args:
            orient str {`dict`, `list`, `records`}: Determines the type of the values of the
                dictionary.

        Returns:
            dict[Scalar, Any]: A dictionary representation of the Series.
        """
        match orient:
            case "dict":
                return self.apply(lambda s: s.to_dict()).data
            case "list":
                return self.apply(lambda s: s.to_list()).data
            case "records":
                return list(self.apply(lambda s: s.to_dict(), axis=1).values)
            case _:
                msg = f"orient '{orient}' not understood"
                raise ValueError(msg)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def _op(
        self, op: str, other: DataFrame | Series | Mapping | ArrayLike | Scalar
    ) -> DataFrame:  # @TODO: Implement axis
        match other:
            case DataFrame():
                return self._op_dataframe(op, other)
            case Series():
                return self._op_series(op, other)
            case ArrayLike() | Mapping() as c if len(c) == 0 and len(self) == 0:
                return DataFrame()
            case ArrayLike() | Mapping() as c if len(c) != len(self.columns):
                msg = f"Unable to coerce to Series, length must be {len(self.columns)}: given {len(other)}"
                raise ValueError(msg)
            case Mapping():
                return self._op_series(op, Series(other))
            case ArrayLike() as c if not isinstance(c, str):
                return self._op_series(op, Series(other, index=self.columns))
            # No 2d collection comparison. Will consider 2d inputs as a series of collections
            case _:  # Everithing else is a scalar then
                return self._op_scalar(op, other)

    def _op_series(self, op: str, other: Series) -> DataFrame:
        if len(self.columns) != len(other):
            msg = "Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating."
            raise ValueError(msg)
        return DataFrame([getattr(row, op)(other) for _, row in self.iterrows()], index=self.index)

    def _op_dataframe(self, op: str, other: DataFrame) -> DataFrame:
        if set(self.keys()) != set(other.keys()):
            msg = "Can only compare identically-labeled (both index and columns) DataFrame objects"
            raise ValueError(msg)
        return DataFrame({col: getattr(s, op)(other[col]) for col, s in self.items()})

    def _op_scalar(self, op: str, other: ArrayLike[Any] | Scalar) -> DataFrame:
        return DataFrame({col: getattr(s, op)(other) for col, s in self.items()})

    def __lt__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__lt__", other)

    def __le__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__le__", other)

    def __eq__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__eq__", other)

    def __ne__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__ne__", other)

    def __gt__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__gt__", other)

    def __ge__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__ge__", other)

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise addition.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__add__", other)

    def __sub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise subtraction.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__sub__", other)

    def __mul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise multiplication.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__mul__", other)

    def __matmul__(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Series by the scalar and returns the sum.

        Args:
            other (Series | ArrayLike | Scalar)

        Returns:
            Scalar: The dot product of the Series.
        """
        return self.dot(other)

    def __truediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise division.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__truediv__", other)

    def __floordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise floor division.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__floordiv__", other)

    def __mod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise modulo.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__mod__", other)

    def __divmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise divmod.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__divmod__", other)

    def __pow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise exponentiation.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__pow__", other)

    def __lshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise left bit shift.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__lshift__", other)

    def __rshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise right bit shift.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__rshift__", other)

    def __and__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise AND.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__and__", other)

    def __xor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise XOR.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__xor__", other)

    def __or__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise OR.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__or__", other)

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    def __radd__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__radd__", other)

    def __rsub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rsub__", other)

    def __rmul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rmul__", other)

    def __rmatmul__(self, other: ArrayLike) -> Series:
        if len(self.index) != len(other):
            msg = f"shapes {Series(other).shape} and {self.shape} not aligned"
            raise ValueError(msg)
        other = Series(other, index=self.index)
        data = [other.dot(s_a) for s_a in self.data.values()]
        return Series(data, index=self.columns, name=other.name)

    def __rtruediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rtruediv__", other)

    def __rfloordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rfloordiv__", other)

    def __rmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rmod__", other)

    def __rdivmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rdivmod__", other)

    def __rpow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rpow__", other)

    def __rlshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rlshift__", other)

    def __rrshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rrshift__", other)

    def __rand__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rand__", other)

    def __rxor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rxor__", other)

    def __ror__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__ror__", other)

    ###########################################################################
    # In-place Operators
    ###########################################################################
    def _iop(self, op: str, other: DataFrame | Series | Mapping | ArrayLike | Scalar) -> Self:
        match other:
            case DataFrame():
                return self._iop_dataframe(op, other)
            case Series():
                return self._iop_series(op, other)
            case ArrayLike() | Mapping() as c if len(c) == 0 and len(self) == 0:
                return self
            case ArrayLike() | Mapping() as c if len(c) != len(self.columns):
                msg = f"Unable to coerce to Series, length must be {len(self.columns)}: given {len(other)}"
                raise ValueError(msg)
            case Mapping():
                return self._iop_series(op, Series(other))
            case ArrayLike() as c if not isinstance(c, str):
                return self._iop_series(op, Series(other, index=self.columns))
            # No 2d collection comparison. Will consider 2d inputs as a series of collections
            case _:  # Everithing else is a scalar then
                return self._iop_scalar(op, other)

    def _iop_series(self, op: str, other: Series) -> Self:
        if len(self.columns) != len(other):
            msg = "Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating."
            raise ValueError(msg)
        for col, s in self.data.items():
            getattr(s, op)(other[col])
        return self

    def _iop_dataframe(self, op: str, other: DataFrame) -> Self:
        if set(self.keys()) != set(other.keys()) or self.shape != other.shape:
            msg = "Can only compare identically-labeled (both index and columns) DataFrame objects"
            raise ValueError(msg)
        for col, s in self.data.items():
            getattr(s, op)(other[col])
        return self

    def _iop_scalar(self, op: str, other: ArrayLike[Any] | Scalar) -> Self:
        for s in self.data.values():
            getattr(s, op)(other)
        return self

    def __iadd__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__iadd__", other)

    def __isub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__isub__", other)

    def __imul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__imul__", other)

    # @TODO: How to overload correctly __matmul__ and __imatmul__?
    # @overload
    # def __imatmul__(self, other: DataFrame) -> DataFrame: ...  # no cov
    # @overload
    # def __imatmul__(self, other: Series | ArrayLike) -> Series: ...  # no cov
    def __imatmul__(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:  # noqa: PYI034
        return self.dot(other)

    def __itruediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__itruediv__", other)

    def __ifloordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ifloordiv__", other)

    def __imod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__imod__", other)

    def __ipow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ipow__", other)

    def __ilshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ilshift__", other)

    def __irshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__irshift__", other)

    def __iand__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__iand__", other)

    def __ixor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ixor__", other)

    def __ior__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:  # type: ignore
        return self._iop("__ior__", other)

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> DataFrame:
        return DataFrame({col: -s for col, s in self.items()})

    def __pos__(self) -> DataFrame:
        return DataFrame({col: +s for col, s in self.items()})

    def __abs__(self) -> DataFrame:
        return self.abs()

    def __invert__(self) -> DataFrame:
        return DataFrame({col: ~s for col, s in self.items()})
