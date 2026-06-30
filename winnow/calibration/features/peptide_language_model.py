"""Peptide language model features for calibration.

This module keeps all heavy PLM dependencies optional. The feature can be used
with precomputed metadata columns in a normal Winnow installation, while local
PepBERT/ESM backends are loaded lazily only when configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Protocol, Sequence
import warnings

import numpy as np
import pandas as pd

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


PLM_MISSING_COLUMN = "is_missing_peptide_language_model_feature"
PLM_LOG_PROB_COLUMN = "plm_mean_log_probability"
PLM_EMBEDDING_COLUMNS = [
    "plm_embedding_mean",
    "plm_embedding_std",
    "plm_embedding_min",
    "plm_embedding_max",
]
PLM_SEQUENCE_COLUMNS = [
    "plm_sequence_length",
    "plm_log_sequence_length",
]
PLM_EMBEDDING_DIAGNOSTIC_COLUMNS = [
    *PLM_EMBEDDING_COLUMNS,
    "plm_pooled_l2_norm",
    "plm_pooled_l2_norm_per_sqrt_length",
    "plm_residue_norm_mean",
    "plm_residue_norm_std",
    "plm_residue_norm_min",
    "plm_residue_norm_max",
    "plm_residue_embedding_std_mean",
    "plm_residue_embedding_std_max",
    "plm_adjacent_cosine_mean",
    "plm_adjacent_cosine_std",
]
PLM_INTERACTION_COLUMNS = [
    "plm_missing_x_log_sequence_length",
]
PLM_EXTENDED_EMBEDDING_COLUMNS = [
    *PLM_SEQUENCE_COLUMNS,
    *PLM_EMBEDDING_DIAGNOSTIC_COLUMNS,
    *PLM_INTERACTION_COLUMNS,
]

PEPBERT_DEFAULT_MODEL = "dzjxzyd/PepBERT-large-UniParc"
ESM2_DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
ESMC_DEFAULT_MODEL = "biohub/ESMC-6B"

_SUPPORTED_BACKENDS = {"precomputed", "pepbert", "esm2", "esmc"}
_SUPPORTED_FEATURE_MODES = {
    "pseudo_likelihood",
    "embedding_summary",
    "embedding_diagnostics",
}
_CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PeptideLanguageModelResult:
    """Compact PLM result for one peptide."""

    mean_log_probability: Optional[float] = None
    embedding_summary: Optional[tuple[float, float, float, float]] = None
    embedding_diagnostics: Optional[tuple[float, ...]] = None


class PeptideLanguageModelBackend(Protocol):
    """Backend interface used by :class:`PeptideLanguageModelFeature`."""

    def score(
        self,
        sequences: Sequence[str],
        feature_mode: str,
    ) -> Dict[str, PeptideLanguageModelResult]:
        """Return PLM results keyed by input sequence."""


def normalize_peptide_for_plm(tokens: object) -> Optional[str]:
    """Normalize Winnow peptide tokens to an unmodified amino-acid string.

    PTM annotations are stripped because peptide/protein language models generally
    expect amino-acid sequences. Tokens that do not resolve to canonical amino acids
    are treated as invalid.
    """
    if tokens is None:
        return None

    if isinstance(tokens, str):
        raw_tokens = _split_string_peptide(tokens)
    elif isinstance(tokens, Iterable):
        raw_tokens = list(tokens)
    else:
        return None

    residues: list[str] = []
    for token in raw_tokens:
        residue = _token_to_base_residue(token)
        if residue is None:
            continue
        if residue not in _CANONICAL_AA:
            return None
        residues.append(residue)

    if not residues:
        return None
    return "".join(residues)


def _split_string_peptide(peptide: str) -> list[str]:
    peptide = peptide.strip()
    if not peptide:
        return []
    if "," in peptide:
        return [part.strip() for part in peptide.split(",") if part.strip()]
    return list(peptide)


def _token_to_base_residue(token: object) -> Optional[str]:
    if not isinstance(token, str):
        return None
    token = token.strip()
    if not token:
        return None
    if token.startswith("[") or token.startswith("("):
        return None

    match = re.match(r"^([A-Z])", token)
    if match is None:
        return None
    return match.group(1)


class HuggingFaceMaskedLMBackend:
    """Generic Hugging Face backend for ESM-like masked language models."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised via tests with monkeypatch
            raise ImportError(
                "Local peptide language model scoring requires optional dependencies "
                "`torch` and `transformers`. Install Winnow with PLM extras or use "
                "backend='precomputed'."
            ) from exc

        self.torch = torch
        self.AutoModel = AutoModel
        self.AutoModelForMaskedLM = AutoModelForMaskedLM
        self.AutoTokenizer = AutoTokenizer
        self.model_name_or_path = model_name_or_path
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._masked_lm_model = None
        self._encoder_model = None

    @property
    def tokenizer(self):
        """Lazily load the backend tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = self.AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer

    @property
    def masked_lm_model(self):
        """Lazily load the masked-language-model head."""
        if self._masked_lm_model is None:
            self._masked_lm_model = self.AutoModelForMaskedLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to(self.device)
            self._masked_lm_model.eval()
        return self._masked_lm_model

    @property
    def encoder_model(self):
        """Lazily load the encoder model."""
        if self._encoder_model is None:
            self._encoder_model = self.AutoModel.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to(self.device)
            self._encoder_model.eval()
        return self._encoder_model

    def score(
        self,
        sequences: Sequence[str],
        feature_mode: str,
    ) -> Dict[str, PeptideLanguageModelResult]:
        """Score sequences with the configured Hugging Face backend."""
        if feature_mode == "pseudo_likelihood":
            return {
                sequence: PeptideLanguageModelResult(
                    mean_log_probability=self._pseudo_likelihood(sequence)
                )
                for sequence in sequences
            }
        if feature_mode == "embedding_summary":
            return {
                sequence: PeptideLanguageModelResult(
                    embedding_summary=self._embedding_summary(sequence)
                )
                for sequence in sequences
            }
        if feature_mode == "embedding_diagnostics":
            return {
                sequence: PeptideLanguageModelResult(
                    embedding_diagnostics=self._embedding_diagnostics(sequence)
                )
                for sequence in sequences
            }
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    def _tokenize(self, sequence: str) -> dict:
        return self.tokenizer(sequence, return_tensors="pt")

    def _special_token_mask(self, input_ids) -> np.ndarray:
        tokenizer = self.tokenizer
        special_ids = {
            token_id
            for token_id in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
                tokenizer.pad_token_id,
            ]
            if token_id is not None
        }
        ids = input_ids[0].detach().cpu().numpy().tolist()
        return np.asarray([token_id in special_ids for token_id in ids], dtype=bool)

    def _pseudo_likelihood(self, sequence: str) -> float:
        tokenizer = self.tokenizer
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                f"Model {self.model_name_or_path!r} tokenizer has no mask token; "
                "pseudo_likelihood mode is unavailable."
            )

        encoded = self._tokenize(sequence)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        special_mask = self._special_token_mask(input_ids)
        positions = [
            idx for idx, is_special in enumerate(special_mask) if not is_special
        ]
        if not positions:
            raise ValueError("No non-special tokens available for pseudo-likelihood.")

        log_probs: list[float] = []
        with self.torch.no_grad():
            for position in positions:
                masked = {
                    key: value.clone() if hasattr(value, "clone") else value
                    for key, value in encoded.items()
                }
                target_id = int(masked["input_ids"][0, position].item())
                masked["input_ids"][0, position] = mask_token_id
                logits = self.masked_lm_model(**masked).logits[0, position]
                token_log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)
                log_probs.append(float(token_log_probs[target_id].detach().cpu()))

        return float(np.mean(log_probs))

    def _token_embeddings(self, sequence: str) -> np.ndarray:
        encoded = self._tokenize(sequence)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        special_mask = self._special_token_mask(input_ids)

        with self.torch.no_grad():
            outputs = self.encoder_model(**encoded)
            hidden = outputs.last_hidden_state[0].detach().cpu().numpy()

        token_embeddings = hidden[~special_mask]
        if len(token_embeddings) == 0:
            raise ValueError("No non-special tokens available for embedding summary.")
        return token_embeddings

    def _embedding_summary(self, sequence: str) -> tuple[float, float, float, float]:
        token_embeddings = self._token_embeddings(sequence)
        pooled = token_embeddings.mean(axis=0)
        return _summarize_embedding(pooled)

    def _embedding_diagnostics(self, sequence: str) -> tuple[float, ...]:
        token_embeddings = self._token_embeddings(sequence)
        pooled = token_embeddings.mean(axis=0)
        return _summarize_embedding_diagnostics(token_embeddings, pooled)


class PepBERTBackend:
    """PepBERT embedding backend using the official Hugging Face files."""

    def __init__(
        self,
        model_name_or_path: str = PEPBERT_DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised via tests with monkeypatch
            raise ImportError(
                "PepBERT scoring requires optional dependencies `torch`, "
                "`huggingface-hub`, and `tokenizers`. Install the PLM extras or use "
                "backend='precomputed'."
            ) from exc

        self.torch = torch
        self.hf_hub_download = hf_hub_download
        self.Tokenizer = Tokenizer
        self.model_name_or_path = model_name_or_path
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        """Lazily load the PepBERT tokenizer."""
        if self._tokenizer is None:
            tokenizer_path = self.hf_hub_download(
                repo_id=self.model_name_or_path,
                filename="tokenizer.json",
                cache_dir=self.cache_dir,
            )
            self._tokenizer = self.Tokenizer.from_file(tokenizer_path)
        return self._tokenizer

    @property
    def model(self):
        """Lazily load the PepBERT encoder model."""
        if self._model is None:
            model_module = self._load_module_from_hub("model.py")
            config_module = self._load_module_from_hub("config.py")
            config = config_module.get_config()
            model = model_module.build_transformer(
                src_vocab_size=self.tokenizer.get_vocab_size(),
                src_seq_len=config["seq_len"],
                d_model=config["d_model"],
            )
            weights_path = self.hf_hub_download(
                repo_id=self.model_name_or_path,
                filename="tmodel_17.pt",
                cache_dir=self.cache_dir,
            )
            state = self.torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state["model_state_dict"])
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def score(
        self,
        sequences: Sequence[str],
        feature_mode: str,
    ) -> Dict[str, PeptideLanguageModelResult]:
        """Score sequences with PepBERT embedding features."""
        if feature_mode == "pseudo_likelihood":
            raise ValueError(
                "PepBERT exposes encoder embeddings in the official release; "
                "use feature_mode='embedding_summary' or backend='precomputed' for "
                "pseudo-likelihood values computed externally."
            )
        if feature_mode not in {"embedding_summary", "embedding_diagnostics"}:
            raise ValueError(f"Unsupported feature_mode for PepBERT: {feature_mode}")
        return self._embedding_summaries(sequences, feature_mode=feature_mode)

    def _load_module_from_hub(self, filename: str):
        file_path = self.hf_hub_download(
            repo_id=self.model_name_or_path,
            filename=filename,
            cache_dir=self.cache_dir,
        )
        module_name = Path(file_path).stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load PepBERT module {filename!r}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _embedding_summary(self, sequence: str) -> tuple[float, float, float, float]:
        summary = self._embedding_summaries(
            [sequence],
            feature_mode="embedding_summary",
        )[sequence].embedding_summary
        if summary is None:
            raise ValueError(
                "PepBERT could not compute an embedding summary for the input sequence."
            )
        return summary

    def _embedding_summaries(
        self,
        sequences: Sequence[str],
        *,
        feature_mode: str,
    ) -> Dict[str, PeptideLanguageModelResult]:
        if len(sequences) == 0:
            return {}

        results = self._empty_results(sequences)
        sos_id, eos_id, pad_id = self._special_token_ids()
        max_model_length = int(self.model.src_pos.seq_len)
        encoded_sequences, too_long, empty = self._encode_sequences(
            sequences,
            sos_id=sos_id,
            eos_id=eos_id,
            max_model_length=max_model_length,
        )

        self._warn_skipped_sequences(
            too_long=too_long,
            empty=empty,
            max_model_length=max_model_length,
        )
        if not encoded_sequences:
            return results

        input_ids, encoder_mask, residue_mask = self._batch_tensors(
            encoded_sequences,
            pad_id=pad_id,
        )

        with self.torch.no_grad():
            embedding = self.model.encode(input_ids, encoder_mask)

        for row_idx, (sequence, _) in enumerate(encoded_sequences):
            residue_embedding = embedding[row_idx][residue_mask[row_idx]]
            pooled = residue_embedding.mean(dim=0).detach().cpu().numpy()
            residue_np = residue_embedding.detach().cpu().numpy()
            if feature_mode == "embedding_diagnostics":
                results[sequence] = PeptideLanguageModelResult(
                    embedding_diagnostics=_summarize_embedding_diagnostics(
                        residue_np,
                        pooled,
                    )
                )
            else:
                results[sequence] = PeptideLanguageModelResult(
                    embedding_summary=_summarize_embedding(pooled)
                )
        return results

    def _empty_results(
        self,
        sequences: Sequence[str],
    ) -> Dict[str, PeptideLanguageModelResult]:
        return {
            sequence: PeptideLanguageModelResult(embedding_summary=None)
            for sequence in sequences
        }

    def _special_token_ids(self) -> tuple[int, int, int]:
        sos_id = self.tokenizer.token_to_id("[SOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        if sos_id is None or eos_id is None:
            raise ValueError("PepBERT tokenizer is missing [SOS] or [EOS] tokens.")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        return sos_id, eos_id, pad_id if pad_id is not None else eos_id

    def _encode_sequences(
        self,
        sequences: Sequence[str],
        *,
        sos_id: int,
        eos_id: int,
        max_model_length: int,
    ) -> tuple[list[tuple[str, list[int]]], int, int]:
        encoded_sequences: list[tuple[str, list[int]]] = []
        too_long = 0
        empty = 0
        for sequence in sequences:
            encoded = [sos_id] + self.tokenizer.encode(sequence).ids + [eos_id]
            if len(encoded) <= 2:
                empty += 1
                continue
            if len(encoded) > max_model_length:
                too_long += 1
                continue
            encoded_sequences.append((sequence, encoded))
        return encoded_sequences, too_long, empty

    def _warn_skipped_sequences(
        self,
        *,
        too_long: int,
        empty: int,
        max_model_length: int,
    ) -> None:
        if too_long:
            warnings.warn(
                f"PepBERT skipped {too_long} sequences longer than its configured "
                f"maximum sequence length ({max_model_length}).",
                UserWarning,
                stacklevel=2,
            )
        if empty:
            warnings.warn(
                f"PepBERT skipped {empty} sequences without residue embeddings.",
                UserWarning,
                stacklevel=2,
            )

    def _batch_tensors(
        self,
        encoded_sequences: list[tuple[str, list[int]]],
        *,
        pad_id: int,
    ):
        max_input_length = max(len(encoded) for _, encoded in encoded_sequences)
        input_ids = self.torch.full(
            (len(encoded_sequences), max_input_length),
            fill_value=pad_id,
            dtype=self.torch.int64,
            device=self.device,
        )
        encoder_mask = self.torch.zeros(
            (len(encoded_sequences), 1, 1, max_input_length),
            dtype=self.torch.int64,
            device=self.device,
        )
        residue_mask = self.torch.zeros(
            (len(encoded_sequences), max_input_length),
            dtype=self.torch.bool,
            device=self.device,
        )
        for row_idx, (_, encoded) in enumerate(encoded_sequences):
            length = len(encoded)
            input_ids[row_idx, :length] = self.torch.tensor(
                encoded,
                dtype=self.torch.int64,
                device=self.device,
            )
            encoder_mask[row_idx, :, :, :length] = 1
            residue_mask[row_idx, 1 : length - 1] = True
        return input_ids, encoder_mask, residue_mask


def _summarize_embedding(embedding: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.mean(embedding)),
        float(np.std(embedding)),
        float(np.min(embedding)),
        float(np.max(embedding)),
    )


def _summarize_embedding_diagnostics(
    residue_embeddings: np.ndarray,
    pooled_embedding: np.ndarray,
) -> tuple[float, ...]:
    length = max(int(residue_embeddings.shape[0]), 1)
    pooled_norm = float(np.linalg.norm(pooled_embedding))
    residue_norms = np.linalg.norm(residue_embeddings, axis=1)
    residue_std = np.std(residue_embeddings, axis=0)

    if length > 1:
        left = residue_embeddings[:-1]
        right = residue_embeddings[1:]
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        adjacent_cosines = np.divide(
            np.sum(left * right, axis=1),
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom > 0,
        )
        adjacent_cosine_mean = float(np.mean(adjacent_cosines))
        adjacent_cosine_std = float(np.std(adjacent_cosines))
    else:
        adjacent_cosine_mean = 0.0
        adjacent_cosine_std = 0.0

    return (
        *_summarize_embedding(pooled_embedding),
        pooled_norm,
        float(pooled_norm / np.sqrt(length)),
        float(np.mean(residue_norms)),
        float(np.std(residue_norms)),
        float(np.min(residue_norms)),
        float(np.max(residue_norms)),
        float(np.mean(residue_std)),
        float(np.max(residue_std)),
        adjacent_cosine_mean,
        adjacent_cosine_std,
    )


class PeptideLanguageModelFeature(CalibrationFeatures):
    """Compute peptide language model features for predicted peptides."""

    def __init__(
        self,
        backend: str = "precomputed",
        model_name_or_path: Optional[str] = None,
        feature_mode: str = "embedding_summary",
        precomputed_column: str = PLM_LOG_PROB_COLUMN,
        batch_size: int = 16,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        learn_from_missing: bool = False,
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported peptide language model backend {backend!r}. "
                f"Supported backends: {sorted(_SUPPORTED_BACKENDS)}."
            )
        if feature_mode not in _SUPPORTED_FEATURE_MODES:
            raise ValueError(
                f"Unsupported feature_mode {feature_mode!r}. "
                f"Supported modes: {sorted(_SUPPORTED_FEATURE_MODES)}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.feature_mode = feature_mode
        self.precomputed_column = precomputed_column
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.learn_from_missing = learn_from_missing
        self._backend_instance: Optional[PeptideLanguageModelBackend] = None
        self._cache: Dict[str, PeptideLanguageModelResult] = {}

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Return feature dependencies required before PLM features."""
        return []

    @property
    def name(self) -> str:
        """Return the display name for this feature."""
        return "Peptide Language Model Feature"

    @property
    def columns(self) -> List[str]:
        """Return metadata columns produced by this feature."""
        if self.feature_mode == "pseudo_likelihood":
            columns = [PLM_LOG_PROB_COLUMN]
        elif self.feature_mode == "embedding_summary":
            columns = list(PLM_EMBEDDING_COLUMNS)
        else:
            columns = list(PLM_EXTENDED_EMBEDDING_COLUMNS)
        if self.learn_from_missing:
            columns.append(PLM_MISSING_COLUMN)
        return columns

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepare the feature before computing values."""
        pass

    def compute(self, dataset: CalibrationDataset) -> None:
        """Compute PLM feature columns on the dataset metadata."""
        if "prediction" not in dataset.metadata.columns:
            raise ValueError(
                "prediction column not found in dataset. "
                "This is required for peptide language model features."
            )

        if self.backend == "precomputed":
            self._compute_precomputed(dataset)
            return

        normalized = dataset.metadata["prediction"].apply(normalize_peptide_for_plm)
        self._handle_missing_inputs(dataset, normalized)
        normalized = dataset.metadata["prediction"].apply(normalize_peptide_for_plm)

        missing = normalized.isna()
        valid_sequences = sorted(set(normalized[~missing].astype(str).tolist()))
        self._score_uncached(valid_sequences)

        if self.feature_mode == "pseudo_likelihood":
            fill_columns = self._compute_pseudo_likelihood_columns(dataset, normalized)
        elif self.feature_mode == "embedding_summary":
            fill_columns = self._compute_embedding_summary_columns(dataset, normalized)
        else:
            fill_columns = self._compute_embedding_diagnostic_columns(
                dataset, normalized
            )

        feature_missing = dataset.metadata[fill_columns].isna().any(axis=1)
        missing = missing | feature_missing
        if self.learn_from_missing:
            dataset.metadata[PLM_MISSING_COLUMN] = missing.astype(bool)
            dataset.metadata.loc[missing, fill_columns] = 0.0
            if self.feature_mode == "embedding_diagnostics":
                dataset.metadata[PLM_INTERACTION_COLUMNS[0]] = (
                    missing.astype(float) * dataset.metadata["plm_log_sequence_length"]
                )
        elif missing.any():
            num_missing = int(missing.sum())
            warnings.warn(
                f"Filtering {num_missing} PSMs with missing peptide language model "
                "features. Set learn_from_missing=True to retain them with imputed "
                "values.",
                UserWarning,
                stacklevel=2,
            )
            _filtered = dataset.filter_entries(
                metadata_predicate=lambda row: bool(missing.loc[row.name])
            )
            dataset.metadata = _filtered.metadata
            dataset.predictions = _filtered.predictions

    def _compute_pseudo_likelihood_columns(
        self,
        dataset: CalibrationDataset,
        normalized: pd.Series,
    ) -> list[str]:
        values = [
            (
                self._cache[sequence].mean_log_probability
                if not pd.isna(sequence)
                else np.nan
            )
            for sequence in normalized
        ]
        dataset.metadata[PLM_LOG_PROB_COLUMN] = values
        return [PLM_LOG_PROB_COLUMN]

    def _compute_embedding_summary_columns(
        self,
        dataset: CalibrationDataset,
        normalized: pd.Series,
    ) -> list[str]:
        summaries: list[tuple[float, float, float, float]] = []
        for sequence in normalized:
            summary = (np.nan, np.nan, np.nan, np.nan)
            if not pd.isna(sequence):
                cached_summary = self._cache[str(sequence)].embedding_summary
                if cached_summary is not None:
                    summary = cached_summary
            summaries.append(summary)

        for column_idx, column in enumerate(PLM_EMBEDDING_COLUMNS):
            dataset.metadata[column] = [summary[column_idx] for summary in summaries]
        return list(PLM_EMBEDDING_COLUMNS)

    def _compute_embedding_diagnostic_columns(
        self,
        dataset: CalibrationDataset,
        normalized: pd.Series,
    ) -> list[str]:
        lengths = normalized.apply(
            lambda sequence: 0.0 if pd.isna(sequence) else float(len(sequence))
        )
        dataset.metadata["plm_sequence_length"] = lengths
        dataset.metadata["plm_log_sequence_length"] = np.log1p(lengths)

        missing_summary = tuple(np.nan for _ in PLM_EMBEDDING_DIAGNOSTIC_COLUMNS)
        summaries: list[tuple[float, ...]] = []
        for sequence in normalized:
            summary = missing_summary
            if not pd.isna(sequence):
                cached_summary = self._cache[str(sequence)].embedding_diagnostics
                if cached_summary is not None:
                    summary = cached_summary
            summaries.append(summary)

        for column_idx, column in enumerate(PLM_EMBEDDING_DIAGNOSTIC_COLUMNS):
            dataset.metadata[column] = [summary[column_idx] for summary in summaries]

        dataset.metadata[PLM_INTERACTION_COLUMNS[0]] = 0.0
        return list(PLM_EMBEDDING_DIAGNOSTIC_COLUMNS)

    def _compute_precomputed(self, dataset: CalibrationDataset) -> None:
        if self.precomputed_column not in dataset.metadata.columns:
            raise ValueError(
                f"Precomputed peptide language model column {self.precomputed_column!r} "
                "not found in dataset metadata."
            )

        values = pd.to_numeric(
            dataset.metadata[self.precomputed_column], errors="coerce"
        )
        missing = values.isna()
        self._handle_missing_inputs(dataset, values.where(~missing, None))

        values = pd.to_numeric(
            dataset.metadata[self.precomputed_column], errors="coerce"
        )
        missing = values.isna()
        if self.feature_mode == "pseudo_likelihood":
            dataset.metadata[PLM_LOG_PROB_COLUMN] = values.astype(float)
            fill_columns = [PLM_LOG_PROB_COLUMN]
        else:
            raise ValueError(
                "Precomputed backend currently supports scalar pseudo-likelihood "
                "columns only. Use feature_mode='pseudo_likelihood'."
            )

        if self.learn_from_missing:
            dataset.metadata[PLM_MISSING_COLUMN] = missing.astype(bool)
            dataset.metadata.loc[missing, fill_columns] = 0.0

    def _handle_missing_inputs(
        self,
        dataset: CalibrationDataset,
        values: pd.Series,
    ) -> None:
        missing = values.isna()
        if not missing.any():
            return
        if self.learn_from_missing:
            return

        num_missing = int(missing.sum())
        warnings.warn(
            f"Filtering {num_missing} PSMs with missing peptide language model "
            "features. Set learn_from_missing=True to retain them with imputed values.",
            UserWarning,
            stacklevel=2,
        )
        _filtered = dataset.filter_entries(
            metadata_predicate=lambda row: bool(missing.loc[row.name])
        )
        dataset.metadata = _filtered.metadata
        dataset.predictions = _filtered.predictions

    def _score_uncached(self, sequences: Sequence[str]) -> None:
        backend = self._get_backend()
        uncached = [sequence for sequence in sequences if sequence not in self._cache]
        for start in range(0, len(uncached), self.batch_size):
            batch = uncached[start : start + self.batch_size]
            self._cache.update(backend.score(batch, self.feature_mode))

    def _get_backend(self) -> PeptideLanguageModelBackend:
        if self._backend_instance is None:
            self._backend_instance = self._build_backend()
        return self._backend_instance

    def _build_backend(self) -> PeptideLanguageModelBackend:
        if self.backend == "pepbert":
            return PepBERTBackend(
                self.model_name_or_path or PEPBERT_DEFAULT_MODEL,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        if self.backend == "esm2":
            return HuggingFaceMaskedLMBackend(
                self.model_name_or_path or ESM2_DEFAULT_MODEL,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        if self.backend == "esmc":
            return HuggingFaceMaskedLMBackend(
                self.model_name_or_path or ESMC_DEFAULT_MODEL,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        raise ValueError(f"Unsupported backend {self.backend!r}.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_backend_instance"] = None
        return state
