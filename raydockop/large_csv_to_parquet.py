from pyarrow import Table
from pyarrow.parquet import ParquetWriter
import pyarrow as pa
import pandas as pd
from pyarrow import csv
class InputStreamReader:
    def __init__(self, file_stream):
        self.file_stream = file_stream
        self._stream = None
    def batches(self):
        i = tries = 0
        while True:
            try:
                batch = self.__next_batch()
                i += 1
                yield i, batch
            except StopIteration:
                break
    def __next_batch(self):
        return self.stream.read_next_batch()
    @property
    def stream(self):
        if not self._stream:
            read_options = pa.csv.ReadOptions(block_size=chunksize)
            parse_options = pa.csv.ParseOptions(delimiter=delimiter)
            convert_options = pa.csv.ConvertOptions(include_columns=include_columns)
            self._stream = pa.csv.open_csv(
                self.file_stream, read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options
            )
        return self._stream

include_columns = ['zincid', 'smiles', 'dockscore']
delimiter = str(',')
chunksize = 1048576*1000
file_stream = '/data/dockop_data/AmpC_screen_table.csv'
input_stream_reader = InputStreamReader(file_stream)

for i, batch in input_stream_reader.batches():
    df = batch.to_pandas()
    table = pa.Table.from_pandas(df)
    schema = table.schema
    print(f'Writing a total of {len(list(df['smiles']))} to disk.')
    ParquetWriter(f'/data/newdockop/dockop/code/mod_code_base/parquet/AmpC_screen_table_part_{i}.parquet', schema).write_table(table)