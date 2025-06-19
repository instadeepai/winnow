import polars as pl

sd = pl.read_parquet(
    "validation_datasets_corrected/spectrum_data/labelled/*_spectrum_all_datasets.parquet"
)
beams = pl.read_csv(
    "validation_datasets_corrected/beam_preds/labelled/*_beam_all_datasets.csv"
)

species = [
    "gluc",
    "helaqc",
    "herceptin",
    "immuno",
    "snakevenoms",
    "sbrodae",
    "woundfluids",
]

for s in species:
    holdout_sd = sd.filter(~(pl.col("source_dataset") == s))
    holdout_beams = beams.filter(~(pl.col("source_dataset") == s))
    holdout_sd.write_parquet(f"holdout_sets/all_less_{s}.parquet")
    holdout_beams.write_csv(f"holdout_sets/all_less_{s}.csv")
