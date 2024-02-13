import click
from compgraph.algorithms import inverted_index_graph


@click.command()
@click.option('--input', help='Input file name')
@click.option('--output', help='Output file name')
@click.option('--doc', default='doc_id', help='Document id column name')
@click.option('--text', default='text', help='Text column name')
@click.option('--res', default='tf_idf', help='Result column name')
def main(input: str, output: str, doc: str, text: str, res: str) -> None:
    graph = inverted_index_graph(input_stream_name=input,
                                 doc_column=doc,
                                 text_column=text,
                                 result_column=res,
                                 filemod=True)

    output_filepath = output

    result = graph.run()
    with open(output_filepath, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
