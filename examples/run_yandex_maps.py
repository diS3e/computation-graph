import click
from compgraph.algorithms import yandex_maps_graph


@click.command()
@click.option('--input-time', help='Input time file name', type=str)
@click.option('--input-length', help='Input length file name', type=str)
@click.option('--output', help='Output file name', type=str)
@click.option('--enter-time', default='enter_time', help='Enter time column name')
@click.option('--leave-time', default='leave_time', help='Leave time column name')
@click.option('--edge-id', default='edge_id', help='Graph edge identifier column name')
@click.option('--start-coord', default='start', help='Start coordinate column name. Coordinate in format("lon", "lat")')
@click.option('--end-coord', default='end', help='End coordinate column name. Coordinate in format("lon", "lat")')
@click.option('--weekday-result', default='weekday', help='Result weekday column name')
@click.option('--hour-result', default='hour', help='Result hour column name')
@click.option('--speed-result', default='speed', help='Result speed column name')
def main(input_time: str, input_length: str, output: str,
         enter_time: str, leave_time: str, edge_id: str,
         start_coord: str, end_coord: str, weekday_result: str, hour_result: str,
         speed_result: str) -> None:
    graph = yandex_maps_graph(input_stream_name_time=input_time,
                              input_stream_name_length=input_length,
                              enter_time_column=enter_time,
                              leave_time_column=leave_time,
                              edge_id_column=edge_id,
                              start_coord_column=start_coord,
                              end_coord_column=end_coord,
                              weekday_result_column=weekday_result,
                              hour_result_column=hour_result,
                              speed_result_column=speed_result,
                              filemod=True)

    output_filepath = output

    result = graph.run()
    with open(output_filepath, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
