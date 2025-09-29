"""Defines data structures for peptide-spectrum matches (PSMs) and datasets.

This module provides classes for handling peptide-spectrum matches, which associate mass spectra with candidate peptide sequences along with confidence scores.
The `PSMDataset` class represents a collection of PSMs and provides utility methods for creating, accessing, and iterating over them.

Classes:
    PeptideSpectrumMatch: Represents a peptide-spectrum match (PSM) with a confidence score.
    PSMDataset: Stores a list of PSMs and provides methods for dataset manipulation.
"""

from dataclasses import dataclass
from typing import List, Sequence, Iterator
from winnow.data_types import Peptide, Spectrum


@dataclass
class PeptideSpectrumMatch:
    """Represents a peptide-spectrum match returned by a peptide sequencing method."""

    spectrum: Spectrum
    peptide: Peptide
    confidence: float

    def __ge__(self, other: "PeptideSpectrumMatch") -> bool:
        """Compare confidence scores between two PSMs.

        Args:
            other (PeptideSpectrumMatch): Another PSM to compare against.

        Returns:
            bool: True if the current PSM has greater or equal confidence than the other.
        """
        return self.confidence >= other.confidence


@dataclass
class PSMDataset:
    """A dataset containing multiple peptide-spectrum matches (PSMs)."""

    peptide_spectrum_matches: List[PeptideSpectrumMatch]

    @classmethod
    def from_dataset(
        cls,
        spectra: Sequence[Spectrum],
        peptides: Sequence[Peptide],
        confidence_scores: Sequence[float],
    ) -> "PSMDataset":
        """Create a PSMDataset from separate sequences of spectra, peptides, and confidence scores.

        Args:
            spectra (Sequence[Spectrum]): List of mass spectra.
            peptides (Sequence[Peptide]): List of peptide sequences.
            confidence_scores (Sequence[float]): Confidence scores for the matches.

        Returns:
            PSMDataset: A dataset containing the constructed PSMs.

        Raises:
            ValueError: If the input sequences don't have the same length.
        """
        # Validate that all sequences have the same length
        lengths = [len(spectra), len(peptides), len(confidence_scores)]
        if len(set(lengths)) > 1:
            raise ValueError("All sequences must have the same length")

        return cls(
            [
                PeptideSpectrumMatch(
                    spectrum=spectrum, peptide=peptide, confidence=confidence
                )
                for spectrum, peptide, confidence in zip(
                    spectra, peptides, confidence_scores
                )
            ]
        )

    def __getitem__(self, index: int) -> PeptideSpectrumMatch:
        """Return the PSM corresponding to an index.

        Args:
            index (int):
                The target index in the dataset.

        Returns:
            PeptideSpectrumMatch:
                The PSM at the index.
        """
        return self.peptide_spectrum_matches[index]

    def __len__(self) -> int:
        """Get the number of PSMs in the dataset.

        Returns:
            int: The total number of PSMs.
        """
        return len(self.peptide_spectrum_matches)

    def __iter__(self) -> Iterator:
        """Iterate over the PSMs in the dataset.

        Returns:
            Iterator: An iterator over the peptide-spectrum matches.
        """
        return iter(self.peptide_spectrum_matches)
