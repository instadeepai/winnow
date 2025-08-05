import polars as pl

sd = pl.read_parquet(
    "validation_datasets_corrected/spectrum_data/labelled/*_spectrum_all_datasets.parquet"
)
beams = pl.read_csv(
    "validation_datasets_corrected/beam_preds/labelled/*_beam_all_datasets.csv"
)

ext_sd = pl.read_parquet(
    "external_datasets/spectrum_data/lcfm/PXD023064/PXD023064.parquet"
)
ext_beams = pl.read_csv("external_datasets/beam_preds/lcfm/PXD023064/PXD023064.csv")

sd_cols_to_keep = [
    "sequence",
    "precursor_mz",
    "precursor_charge",
    "precursor_mass",
    "retention_time",
    "mz_array",
    "intensity_array",
    "spectrum_id",
]

ext_sd = ext_sd.select(sd_cols_to_keep).with_columns(
    pl.lit("PXD023064").alias("source_dataset")
)
sd = sd.select(sd_cols_to_keep + ["source_dataset"])

combined_sd = pl.concat([sd, ext_sd], how="vertical_relaxed")

beams_cols_to_keep = ext_beams.columns[3:]

beams = beams.select(beams_cols_to_keep + ["source_dataset"])
ext_beams = ext_beams.select(beams_cols_to_keep).with_columns(
    pl.lit("PXD023064").alias("source_dataset")
)
combined_beams = pl.concat([beams, ext_beams], how="vertical_relaxed")

species = [
    "gluc",
    "helaqc",
    "herceptin",
    "immuno",
    "snakevenoms",
    "sbrodae",
    "woundfluids",
    "PXD023064",
]

for s in species:
    holdout_sd = sd.filter(~(pl.col("source_dataset") == s))
    holdout_beams = beams.filter(~(pl.col("source_dataset") == s))
    holdout_sd.write_parquet(f"holdout_sets/all_less_{s}.parquet")
    holdout_beams.write_csv(f"holdout_sets/all_less_{s}.csv")
