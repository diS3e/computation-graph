import click
from compgraph.algorithms import word_count_graph


@click.command()
@click.option('--input', help='Input file name')
@click.option('--output', help='Output file name')
@click.option('--text', default='text', help='Text column name')
@click.option('--count', default='count', help='Result count column name')
def main(input: str, output: str, text: str, count: str) -> None:
    graph = word_count_graph(input_stream_name=input,
                             text_column=text,
                             count_column=count,
                             filemod=True)

    output_filepath = output

    result = graph.run()
    with open(output_filepath, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
