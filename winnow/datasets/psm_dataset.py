from dataclasses import dataclass
from typing import List, Sequence
from winnow.types import (
    Peptide,
    Spectrum
)

@dataclass
class PeptideSpectrumMatch:
    """Represents a peptide-spectrum match returned by a peptide sequencing method.
    """
    spectrum: Spectrum
    peptide: Peptide
    confidence: float

    def __ge__(self, other: "PeptideSpectrumMatch") -> bool:
        return self.confidence >= other.confidence


@dataclass
class PSMDataset:
    peptide_spectrum_matches: List[PeptideSpectrumMatch]

    @classmethod
    def from_dataset(
        cls, 
        spectra: Sequence[Spectrum],
        peptides: Sequence[Peptide],
        confidence_scores: Sequence[float]
    ) -> "PSMDataset":
        return cls(
            [
                PeptideSpectrumMatch(
                    spectrum=spectrum,
                    peptide=peptide,
                    confidence=confidence
                )
                for spectrum, peptide, confidence in zip(spectra, peptides, confidence_scores)
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
        return len(self.peptide_spectrum_matches)
