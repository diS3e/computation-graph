import copy
import typing as tp
from .external_sort import ExternalSort
from . import operations as ops


class Graph:
    """Computational graph implementation"""

    def __init__(self, factory: ops.Operation) -> None:
        self.__stream: ops.Operation = factory
        self.__op_list: list[tp.Callable[[ops.TRowsIterable], ops.TRowsGenerator]] = list()

    def __add_operation(self, operation: tp.Callable[[ops.TRowsIterable], ops.TRowsGenerator]) -> 'Graph':
        cop = copy.deepcopy(self)
        cop.__op_list.append(operation)
        return cop

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        return Graph(ops.ReadIterFactory(name))

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(ops.Read(filename, parser))

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        return self.__add_operation(ops.Map(mapper))

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return self.__add_operation(ops.Reduce(reducer, keys))

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return self.__add_operation(ExternalSort(keys))

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """

        def wrap(rows: ops.TRowsIterable, **kwargs: dict[str, tp.Any]) -> ops.TRowsGenerator:
            return ops.Join(joiner, keys)(rows, join_graph.run(**kwargs))

        return self.__add_operation(wrap)

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        res = self.__stream(**kwargs)
        for op in self.__op_list:
            res = op(res, **kwargs)
        return res
