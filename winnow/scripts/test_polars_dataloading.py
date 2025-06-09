import time
import os
import ray
import polars as pl
from winnow.calibration.training.data import pad_batch_spectra
from typing import Callable

class RayDataLoader:
    def __init__(self, df: pl.DataFrame, collect_fn: Callable):
        self.df = df
        self.data_length = df.select(pl.len()).item()
        self.collect_fn = collect_fn
        
        def process_batch_ray(df: pl.DataFrame, batch_index: int, batch_size: int, num_peaks: int):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            batch = df.slice(batch_index, batch_size).to_dicts()
            keys = batch[0].keys()
            return collect_fn({
                key: [sample[key] for sample in batch]
                for key in keys
            }, num_peaks=num_peaks)
        self.process_batch = ray.remote(process_batch_ray)

    def iter_batches(self, batch_size: int, num_peaks: int):
        ray_df = ray.put(self.df)
        tasks = [self.process_batch.remote(ray_df, batch_index, batch_size, num_peaks)
                 for batch_index in range(0, self.data_length, batch_size)]
        while tasks:
            ready, tasks = ray.wait(tasks)
            for ready_task in ready:
                yield ray.get(ready_task)






df = pl.scan_parquet(
    "s3://winnow-g88rh/transformer/val.parquet",
    storage_options={
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
    }
).limit(1600).collect()

ray.init()


def process_batch(batch: dict, num_peaks: int):
    return pad_batch_spectra({
        key: [sample[key] for sample in batch]
        for key in batch[0].keys()
    }, num_peaks=num_peaks)


# @ray.remote
# def process_batch_ray(df: pl.DataFrame, batch_index: int, batch_size: int, num_peaks: int):
#     batch = df.slice(batch_index, batch_size).collect().to_dicts()
#     return pad_batch_spectra({
#         key: [sample[key] for sample in batch]
#         for key in batch[0].keys()
#     }, num_peaks=num_peaks)

batch_size = 10

print('Ray timing')
t_start = time.perf_counter()
dataloader = RayDataLoader(df, pad_batch_spectra)
t_init = time.perf_counter()
print(f"Time taken to initialize dataloader: {t_init - t_start}")
t = t_init
for batch in dataloader.iter_batches(batch_size=batch_size, num_peaks=200):
    t_next = time.perf_counter()
    print(f"Time taken: {t_next - t}")
    t = t_next
print(f"Total time taken: {time.perf_counter() - t_start}")

print('Polars timing')
data_length = df.select(pl.len()).item()
t_start = time.perf_counter()
t = t_start
for batch_index in range(0, data_length, batch_size):
    process_batch(df.slice(batch_index, batch_size).to_dicts(), num_peaks=200)
    t_next = time.perf_counter()
    print(f"Time taken: {t_next - t}")
    t = t_next
print(f"Total time taken: {time.perf_counter() - t_start}")