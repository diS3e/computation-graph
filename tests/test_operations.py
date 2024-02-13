import copy
import dataclasses
import typing as tp

import pytest

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


def tf_idf(row: ops.TRow) -> tp.Iterable[ops.TRow]:
    return [row | {'tf_idf': row['tf'] * row['idf']}]


MAP_CASES = [
    MapCase(
        mapper=ops.LambdaMapper(tf_idf),
        data=[
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.25},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 5,
             'tf': 2 * 0.3333333333333333},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 6, 'tf': 0.2},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 2, 'tf': 1.0},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 3, 'tf': 1.0},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.5},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.25},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 5,
             'tf': 0.3333333333333333},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 6, 'tf': 0.8}
        ],
        ground_truth=[
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333, 'tf_idf': 0.13515503603605478},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.25,
             'tf_idf': 0.1013662770270411},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 5,
             'tf': 0.6666666666666666, 'tf_idf': 0.27031007207210955},
            {'text': 'hello', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 6, 'tf': 0.2,
             'tf_idf': 0.08109302162163289},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333, 'tf_idf': 0.13515503603605478},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 2, 'tf': 1.0,
             'tf_idf': 0.4054651081081644},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 3, 'tf': 1.0,
             'tf_idf': 0.4054651081081644},
            {'text': 'little', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.5,
             'tf_idf': 0.2027325540540822},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 1,
             'tf': 0.3333333333333333, 'tf_idf': 0.13515503603605478},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 4, 'tf': 0.25,
             'tf_idf': 0.1013662770270411},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 5,
             'tf': 0.3333333333333333, 'tf_idf': 0.13515503603605478},
            {'text': 'world', 'doc_count': 4, 'size': 6, 'idf': 0.4054651081081644, 'doc_id': 6, 'tf': 0.8,
             'tf_idf': 0.32437208648653154}
        ],
        cmp_keys=('doc_id', 'text', 'tf_idf')
    ),
    MapCase(
        mapper=ops.Haversine('start', 'end', 'result'),
        data=[
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953]},
            {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824]},
            {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356]},
            {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035]},
            {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032]},
            {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584]},
            {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706]}
        ],
        ground_truth=[
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
             'result': 0.032023888626150986},
            {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
             'result': 0.04546412435707955},
            {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
             'result': 0.035647822076899226},
            {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
             'result': 0.04118458581562971},
            {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
             'result': 0.12515660131114523},
            {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
             'result': 0.006075476856837266},
            {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
             'result': 0.004089360693442873}
        ],
        cmp_keys=('start', 'end', 'result')
    )
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.Mean(words_column='count'),
        reducer_keys=('doc_id', 'text'),
        data=[
            {'doc_id': 1, 'text': 'hello', 'count': 1},
            {'doc_id': 1, 'text': 'little', 'count': 3},
            {'doc_id': 1, 'text': 'world', 'count': 5},

            {'doc_id': 2, 'text': 'little', 'count': 1},

            {'doc_id': 3, 'text': 'little', 'count': 13.1},
            {'doc_id': 3, 'text': 'little', 'count': 2.9},
            {'doc_id': 3, 'text': 'little', 'count': 5},

            {'doc_id': 4, 'text': 'little', 'count': 22},
            {'doc_id': 4, 'text': 'hello', 'count': 13},
            {'doc_id': 4, 'text': 'little', 'count': 12},
            {'doc_id': 4, 'text': 'world', 'count': 11.11},

            {'doc_id': 5, 'text': 'hello', 'count': 1},
            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'world', 'count': 9},

            {'doc_id': 6, 'text': 'world', 'count': -100},
            {'doc_id': 6, 'text': 'world', 'count': -1},
            {'doc_id': 6, 'text': 'world', 'count': 13},
            {'doc_id': 6, 'text': 'world', 'count': 23},
            {'doc_id': 6, 'text': 'hello', 'count': 78}
        ],
        ground_truth=[
            {'doc_id': 1, 'mean': 1.0, 'text': 'hello'},
            {'doc_id': 1, 'mean': 3.0, 'text': 'little'},
            {'doc_id': 1, 'mean': 5.0, 'text': 'world'},

            {'doc_id': 2, 'mean': 1.0, 'text': 'little'},

            {'doc_id': 3, 'mean': 7.0, 'text': 'little'},

            {'doc_id': 4, 'mean': 13.0, 'text': 'hello'},
            {'doc_id': 4, 'mean': 22.0, 'text': 'little'},
            {'doc_id': 4, 'mean': 12.0, 'text': 'little'},
            {'doc_id': 4, 'mean': 11.11, 'text': 'world'},

            {'doc_id': 5, 'mean': 1.5, 'text': 'hello'},
            {'doc_id': 5, 'mean': 9.0, 'text': 'world'},

            {'doc_id': 6, 'mean': 78.0, 'text': 'hello'},
            {'doc_id': 6, 'mean': -16.25, 'text': 'world'}
        ],
        cmp_keys=('doc_id', 'text', 'mean'),
        reduce_data_items=(0,),
        reduce_ground_truth_items=(0,)
    )
]


@pytest.mark.parametrize('case', REDUCE_CASES)
def test_reducer(case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(case.data[i]) for i in case.reduce_data_items]
    reducer_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.reduce_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    reducer_result = case.reducer(case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_result, key=key_func) == sorted(reducer_ground_truth_rows, key=key_func)

    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)
