import ast
from math import log, e
from dateutil.parser import parse  # type: ignore
import typing as tp
from .graph import Graph
import compgraph.operations as ops


def iter_or_file(filemod: bool, input_name: str, parser: tp.Callable[[str], ops.TRow]) -> Graph:
    if filemod:
        return Graph.graph_from_file(input_name, parser)
    else:
        return Graph.graph_from_iter(input_name)


def word_count_graph(input_stream_name: str, text_column: str = 'text',
                     count_column: str = 'count',
                     *, filemod: bool = False, parser: tp.Callable[[str], ops.TRow] = ast.literal_eval) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return iter_or_file(filemod, input_stream_name, parser) \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column)) \
        .sort([text_column]) \
        .reduce(ops.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf',
                         *, filemod: bool = False, parser: tp.Callable[[str], ops.TRow] = ast.literal_eval) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    input_stream = iter_or_file(filemod, input_stream_name, parser)
    split_word = input_stream \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column))

    count_docs = input_stream \
        .reduce(ops.Count("size"), [])

    def idf(row: ops.TRow) -> ops.TRowsIterable:
        return [row | {'idf': log(row['size'] / row['doc_count'], e)}]

    count_idf = split_word \
        .sort([doc_column, text_column]) \
        .reduce(ops.TopN(doc_column, 1), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(ops.Count('doc_count'), [text_column]) \
        .join(ops.InnerJoiner(), count_docs, []) \
        .map(ops.LambdaMapper(idf)) \
        .sort([text_column])

    tf = split_word \
        .sort([doc_column]) \
        .reduce(ops.TermFrequency(text_column), [doc_column]) \
        .sort([text_column])

    def tf_idf(row: ops.TRow) -> ops.TRowsIterable:
        return [row | {result_column: row['tf'] * row['idf']}]

    return count_idf \
        .join(ops.InnerJoiner(), tf, [text_column]) \
        .map(ops.LambdaMapper(tf_idf)) \
        .sort([text_column]) \
        .reduce(ops.TopN(result_column, 3), [text_column]) \
        .map(ops.Project([doc_column, text_column, result_column])) \
        .sort([doc_column, text_column])


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi',
              *, filemod: bool = False, parser: tp.Callable[[str], ops.TRow] = ast.literal_eval) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    suffix_a = '_1'
    suffix_b = '_2'
    joiner = ops.InnerJoiner(suffix_a, suffix_b)
    count_column = 'count_in_doc'
    frequency_column = 'frequency'
    split_word = iter_or_file(filemod, input_stream_name, parser) \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column)) \
        .map(ops.Filter(lambda x: len(x[text_column]) > 4)) \
        .sort([doc_column, text_column]) \
        .reduce(ops.Count(count_column), [doc_column, text_column]) \
        .map(ops.Filter(lambda x: x[count_column] >= 2))

    word_count = split_word \
        .sort([text_column]) \
        .reduce(ops.Sum(count_column), [text_column])

    all_words = split_word \
        .reduce(ops.Sum(count_column), [])
    words_in_doc = split_word \
        .sort([doc_column]) \
        .reduce(ops.Sum(count_column), [doc_column])

    def freq(size_col: str, count_col: str) -> tp.Callable[[ops.TRow], tp.Iterable[ops.TRow]]:
        return lambda row: [row | {frequency_column: row[count_col] / row[size_col]}]

    in_all = all_words \
        .join(joiner, word_count, []) \
        .map(ops.LambdaMapper(freq(count_column + suffix_a, count_column + suffix_b))) \
        .sort([text_column])

    in_doc = words_in_doc \
        .join(ops.InnerJoiner(), split_word, [doc_column]) \
        .map(ops.LambdaMapper(freq(count_column + suffix_a, count_column + suffix_b))) \
        .sort([text_column])

    def ln(row: ops.TRow) -> ops.TRowsIterable:
        return [row | {result_column: log(row[frequency_column], e)}]

    res = in_all \
        .join(joiner, in_doc, [text_column]) \
        .map(ops.LambdaMapper(freq(frequency_column + suffix_a, frequency_column + suffix_b))) \
        .map(ops.LambdaMapper(ln)) \
        .sort([doc_column]) \
        .reduce(ops.TopN(result_column, 10), [doc_column]) \
        .map(ops.Project([doc_column, result_column, text_column]))
    return res


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed',
                      *, filemod: bool = False, parser: tp.Callable[[str], ops.TRow] = ast.literal_eval) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""

    delta_time_column = "delta"
    road_length_column = "length"

    def time_delta(row: ops.TRow) -> ops.TRowsIterable:
        start = row[enter_time_column]
        end = row[leave_time_column]
        dt_start = parse(start)
        dt_end = parse(end)
        start_parsed = dt_start.date().strftime('%a'), dt_start.time().hour
        delta_t = (dt_end - dt_start).total_seconds() / 3600
        return [row | {
            weekday_result_column: start_parsed[0],
            hour_result_column: start_parsed[1],
            delta_time_column: delta_t
        }]

    time_input = iter_or_file(filemod, input_stream_name_time, parser) \
        .map(ops.LambdaMapper(time_delta)) \
        .sort([edge_id_column])

    length_input = iter_or_file(filemod, input_stream_name_length, parser) \
        .map(ops.Haversine(start_coord_column, end_coord_column, road_length_column)) \
        .sort([edge_id_column])

    def mean_speed(row: ops.TRow) -> ops.TRowsIterable:
        dist = row[road_length_column]
        time = row[delta_time_column]
        return [row | {speed_result_column: dist / time}]

    return time_input \
        .join(ops.InnerJoiner(), length_input, [edge_id_column]) \
        .map(ops.LambdaMapper(mean_speed)) \
        .map(ops.Project([weekday_result_column, hour_result_column, speed_result_column])) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(ops.Mean(speed_result_column, speed_result_column),
                [weekday_result_column, hour_result_column])
