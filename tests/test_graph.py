import ast
import tempfile

from compgraph.graph import Graph
from compgraph import operations as ops

SIMPLE_TABLE = [
    {'doc_id': 1, 'text': 'hello, little world'},
    {'doc_id': 2, 'text': 'little'},
    {'doc_id': 3, 'text': 'little little little'},
    {'doc_id': 4, 'text': 'little? hello little world'},
    {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
    {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
]


def test_constructor_graph() -> None:
    graph = Graph(ops.ReadIterFactory('texts')).map(ops.DummyMapper())
    result = graph.run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == SIMPLE_TABLE


def test_graph_from_iter_without_commands() -> None:
    graph = Graph.graph_from_iter('texts')
    result = graph.run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == SIMPLE_TABLE


def test_graph_from_file_without_commands() -> None:
    file = tempfile.NamedTemporaryFile()
    with open(file.name, 'w') as f:
        for row in SIMPLE_TABLE:
            print(row, file=f)

    graph = Graph.graph_from_file(file.name, ast.literal_eval)
    result = graph.run()
    assert list(result) == SIMPLE_TABLE
    file.close()


def test_graph_from_iter_with_dummy_mapper() -> None:
    graph = Graph.graph_from_iter('texts').map(ops.DummyMapper())
    result = graph.run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == SIMPLE_TABLE


def test_graph_from_file_with_dummy_mapper() -> None:
    file = tempfile.NamedTemporaryFile()
    with open(file.name, 'w') as f:
        for row in SIMPLE_TABLE:
            print(row, file=f)

    graph = (Graph.graph_from_file(file.name, ast.literal_eval)
             .map(ops.DummyMapper()))
    result = graph.run()
    assert list(result) == SIMPLE_TABLE
    file.close()


def test_graph_first_reduce() -> None:
    graph = Graph.graph_from_iter('texts').reduce(ops.FirstReducer(), ['doc_id'])
    result = graph.run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == SIMPLE_TABLE

    graph = Graph.graph_from_iter('texts').reduce(ops.FirstReducer(), [])
    result = graph.run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == [SIMPLE_TABLE[0]]


def test_graph_inner_join_with_non_linear() -> None:
    graph = Graph.graph_from_iter('texts')
    graph1 = graph.map(ops.FilterPunctuation('text'))
    graph2 = graph.map(ops.LowerCase('text'))
    result = (graph1.join(ops.InnerJoiner(), graph2, ['doc_id'])).run(texts=lambda: iter(SIMPLE_TABLE))
    assert list(result) == [
        {'doc_id': 1, 'text_1': 'hello little world', 'text_2': 'hello little world'},
        {'doc_id': 2, 'text_1': 'little', 'text_2': 'little'},
        {'doc_id': 3, 'text_1': 'little little little', 'text_2': 'little little little'},
        {'doc_id': 4, 'text_1': 'little hello little world', 'text_2': 'little hello little world'},
        {'doc_id': 5, 'text_1': 'hello hello world', 'text_2': 'hello hello world'},
        {'doc_id': 6, 'text_1': 'world world world world hello', 'text_2': 'world world world world hello'}
    ]
