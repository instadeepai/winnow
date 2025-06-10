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
        t = time.perf_counter()
        ray_df = ray.put(self.df)
        print(f"Time taken to put df: {time.perf_counter() - t}")
        t = time.perf_counter()
        tasks = [self.process_batch.remote(ray_df, batch_index, batch_size, num_peaks)
                 for batch_index in range(0, self.data_length, batch_size)]
        print(f"Time taken to create tasks: {time.perf_counter() - t}")
        t = time.perf_counter()
        while tasks:
            ready, tasks = ray.wait(tasks)
            print(f"Time taken to wait: {time.perf_counter() - t}")
            t = time.perf_counter()
            for ready_task in ready:
                t = time.perf_counter()
                batch = ray.get(ready_task)
                print(f"Time taken to get batch: {time.perf_counter() - t}")
                t = time.perf_counter()
                yield batch


@ray.remote
class BatchProcessor:
    def __init__(self, batch_size: int, num_peaks: int, collect_fn: Callable):
        self.batch_size = batch_size
        self.num_peaks = num_peaks
        self.collect_fn = collect_fn
        
    def process_batch(self, df: pl.DataFrame, batch_index: int):
        batch = df.slice(batch_index, self.batch_size).to_dicts()
        return self.collect_fn({
            key: [sample[key] for sample in batch]
            for key in batch[0].keys()
        }, num_peaks=self.num_peaks)

class RayActorDataLoader:
    def __init__(self, df: pl.DataFrame, collect_fn: Callable, num_actors: int):
        self.df = df
        self.collect_fn = collect_fn
        self.num_actors = num_actors
        
    def iter_batches(self, batch_size: int, num_peaks: int):
        t = time.perf_counter()
        data_length = self.df.select(pl.len()).item()
        print(f"Time taken to get data length: {time.perf_counter() - t}")
        t = time.perf_counter()
        df_ref = ray.put(self.df)
        print(f"Time taken to put df: {time.perf_counter() - t}")
        t = time.perf_counter()
        actor_pool = ray.util.ActorPool([
            BatchProcessor.remote(batch_size, num_peaks, self.collect_fn)
            for _ in range(self.num_actors)
        ])
        print(f"Time taken to create actor pool: {time.perf_counter() - t}")
        t = time.perf_counter()
        return actor_pool.map_unordered(
            lambda actor, batch_index: actor.process_batch.remote(df_ref, batch_index),
            range(0, data_length, batch_size)
        )


df = pl.scan_parquet(
    "s3://winnow-g88rh/transformer/val.parquet",
    storage_options={
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
    }
).limit(400).collect()

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
dataloader = RayActorDataLoader(df, pad_batch_spectra, num_actors=4)
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