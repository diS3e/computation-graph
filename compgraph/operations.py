import heapq
from abc import abstractmethod, ABC
import typing as tp
import re
import itertools
from collections import defaultdict
from math import radians, sin, cos, asin, sqrt

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for i in rows:
            yield from self.mapper(i)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for value, group in itertools.groupby(rows, key=lambda x: tuple(x[i] for i in self.keys)):
            yield from self.reducer(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    def _abs_joiner(self, a_empty: bool, b_empty: bool, keys: tp.Sequence[str],
                    rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        mat_rows_b = list(rows_b)
        if len(mat_rows_b) == 0 and a_empty:
            yield from rows_a
        flag_b = True
        for a_row in rows_a:
            for b_row in mat_rows_b:
                flag_b = False
                if all(a_row[i] == b_row[i] for i in keys):
                    row = dict()
                    for key, value in (a_row | b_row).items():
                        if key in a_row and key in b_row and key not in keys:
                            row[key + self._a_suffix] = a_row[key]
                            row[key + self._b_suffix] = b_row[key]
                        else:
                            row[key] = value
                    yield row

        if flag_b and b_empty:
            yield from mat_rows_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        other_rows = args[0]
        rows_iter = itertools.groupby(rows, lambda x: tuple(x[i] for i in self.keys))
        other_rows_iter = itertools.groupby(other_rows, lambda x: tuple(x[i] for i in self.keys))
        try:
            for (value, group) in rows_iter:
                for (other_value, other_group) in other_rows_iter:
                    if value > other_value:
                        yield from self.joiner(self.keys, [], other_group)
                        continue
                    while value < other_value:
                        yield from self.joiner(self.keys, group, [])
                        value, group = next(rows_iter)
                    if value == other_value:
                        yield from self.joiner(self.keys, group, other_group)
                else:
                    for _, i in rows_iter:
                        yield from self.joiner(self.keys, i, [])
        except StopIteration:
            yield from self.joiner(self.keys, [], other_group)
        finally:
            for _, i in other_rows_iter:
                yield from self.joiner(self.keys, [], i)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers

class LambdaMapper(Mapper):
    """Map rows using passed function"""
    def __init__(self, function: tp.Callable[[tp.Dict[str, tp.Any]], tp.Iterable[tp.Dict[str, tp.Any]]]):
        """
        :param function: function to map row. Take row, return iterable of rows
        """
        self.f = function

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = self.f(row)
        for i in res:
            yield i


class Haversine(Mapper):
    """Find distance between two points using haversine function"""
    def __init__(self, start_coord_column: str, end_coord_column: str, result_column: str):
        """
        :param start_coord_column: name of column with start point ('lon', 'lat')
        :param end_coord_column: name of column with end point ('lon', 'lat')
        :param result_column: name of result column
        """
        self.start = start_coord_column
        self.end = end_coord_column
        self.result = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        r = 6373
        lambda1, phi1 = row[self.start]
        lambda2, phi2 = row[self.end]
        lambda1, phi1 = radians(lambda1), radians(phi1)
        lambda2, phi2 = radians(lambda2), radians(phi2)
        h = ((sin((phi2 - phi1) / 2)) ** 2) + (cos(phi1) * cos(phi2) * ((sin((lambda2 - lambda1) / 2)) ** 2))
        yield row | {self.result: 2 * r * asin(sqrt(h))}


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = str.translate(row[self.column],
                                         str.maketrans('', '', r'!“”"‘’#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = LowerCase._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = r"\s+") -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        for match in re.finditer(f'(?:^|{self.separator})((?:(?!{self.separator}).)*)', row[self.column]):
            fragment = match.group(1)
            yield row.copy() | {self.column: fragment}


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans = 1
        for i in self.columns:
            ans *= row[i]
        row[self.result_column] = ans
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key: row[key] for key in self.columns}


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        heap: list[tuple[tp.Any, tp.ItemsView[str, tp.Any]]] = []
        for row in rows:
            if len(heap) < self.n:
                heapq.heappush(heap, (row[self.column_max], row.items()))
            else:
                heapq.heappushpop(heap, (row[self.column_max], row.items()))

        arr = []
        while len(heap) != 0:
            arr.append(heapq.heappop(heap)[1])
        for i in reversed(arr):
            yield dict(i)


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        size = 0
        value_dict: dict[str, int] = defaultdict(int)
        main_part = dict()
        for value, group in itertools.groupby(rows,
                                              key=lambda x: x[self.words_column]):
            group_size = 0
            for row in group:
                group_size += 1
                main_part = row
            size += group_size
            value_dict[value] += group_size

        for value, group_size in value_dict.items():
            yield ({key: main_part[key] for key in group_key} |
                   {self.result_column: group_size / size} |
                   {self.words_column: value})


class Mean(Reducer):
    """Calculate mean of values in column"""

    def __init__(self, words_column: str, result_column: str = 'mean') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        size = 0
        main_part = dict()
        group_sum = 0
        for row in rows:
            main_part = row
            group_sum += row[self.words_column]
            size += 1
        yield {key: main_part[key] for key in group_key} | {self.result_column: group_sum / size}


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        size = 0
        value = dict()
        for row in rows:
            size += 1
            value = row
        yield {key: value[key] for key in group_key} | {self.column: size}


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        size = 0
        value = dict()
        for row in rows:
            size += row[self.column]
            value = row
        yield {key: value[key] for key in group_key} | {self.column: size}


# Joiners
class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self._abs_joiner(False, False, keys, rows_a, rows_b)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self._abs_joiner(True, True, keys, rows_a, rows_b)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self._abs_joiner(True, False, keys, rows_a, rows_b)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self._abs_joiner(False, True, keys, rows_a, rows_b)
