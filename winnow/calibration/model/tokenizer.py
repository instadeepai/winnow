import re
import typing as tp
import jax.numpy as jnp


class ModifiedPeptideTokenizer:
    """Tokenizes and detokenizes modified peptides with residue and modification vocabularies."""

    def __init__(self, residues: tp.Tuple[str, ...], modifications: tp.Tuple[str, ...]):
        """Initialize the tokenizer with residue and modification vocabularies.

        Args:
            residues: Tuple of single-letter amino acid codes (e.g., ('A', 'C', 'D', ...)).
            modifications: Tuple of modification identifiers (e.g., ('[UNIMOD:4]', '[UNIMOD:35]', ...)).
        """
        # Define vocabularies
        self.residue_vocabulary = residues + ("", "[PAD]")
        self.residue_pad_index = len(self.residue_vocabulary) - 1

        self.modifications_vocabulary = modifications + ("", "[PAD]")
        self.modification_pad_index = len(self.modifications_vocabulary) - 1

        # Define index mappings
        self.residues_to_indices = {
            residue: index for index, residue in enumerate(self.residue_vocabulary)
        }
        self.modifications_to_indices = {
            modification: index
            for index, modification in enumerate(self.modifications_vocabulary)
        }

    def tokenize(self, peptide: str) -> tuple[list[int], list[int]]:
        """Tokenize a modified peptide into residue and modification indices.

        Args:
            peptide: A string representing the modified peptide (e.g., 'ACDEFGHIK[UNIMOD:35]').

        Returns:
            A tuple of two lists: residue indices and modification indices.
            Residue indices correspond to the amino acid residues, and modification indices
            correspond to the modifications (or empty string if no modification).
        """
        residue_indices, modification_indices = [], []
        i = 0
        while i < len(peptide):
            residue = ""
            modification = ""

            # Case 1: Amino Acid Residue
            if "A" <= peptide[i] <= "Z":
                residue = peptide[i]
                i += 1
                # Check for a following modification
                if i < len(peptide) and peptide[i] == "[":
                    match = re.match(r"(\[.+?\])", peptide[i:])
                    if match:
                        modification = match.group(1)
                        i += len(modification)

            # Case 2: Modification without a preceding residue
            elif peptide[i] == "[":
                match = re.match(r"(\[.+?\])", peptide[i:])
                if match:
                    # Residue is empty string
                    modification = match.group(1)
                    i += len(modification)
                else:
                    # Not a valid modification tag, raise an exception
                    raise ValueError(
                        f"Invalid peptide sequence: unmatched '[' at position {i} in '{peptide}'"
                    )
            else:
                # Unrecognized character, raise an exception
                raise ValueError(
                    f"Invalid character in peptide: '{peptide[i]}' at position {i} in '{peptide}'"
                )

            try:
                residue_indices.append(self.residues_to_indices[residue])
            except KeyError:
                print(
                    f"Residue '{residue}' not found in vocabulary. Sequence: {peptide}"
                )
                residue_indices.append(self.residue_pad_index)

            try:
                modification_indices.append(
                    self.modifications_to_indices[modification]
                )
            except KeyError:
                print(
                    f"Modification '{modification}' not found in vocabulary. Sequence: {peptide}"
                )
                modification_indices.append(self.modifications_to_indices[""])

        assert len(residue_indices) == len(modification_indices)
        return residue_indices, modification_indices

    def detokenize(
        self, residue_indices: list[int], modification_indices: list[int]
    ) -> str:
        """Detokenize residue and modification indices into a modified peptide string.

        Args:
            residue_indices: List of indices corresponding to amino acid residues.
            modification_indices: List of indices corresponding to modifications.

        Returns:
            A string representing the modified peptide (e.g., 'ACDEFGHIK[UNIMOD:35]').
        """
        assert len(residue_indices) == len(
            modification_indices
        ), "Residue and modification indices must have the same length."
        assert all(
            0 <= index < len(self.residue_vocabulary) for index in residue_indices
        ), "Invalid residue index."
        assert all(
            0 <= index < len(self.modifications_vocabulary)
            for index in modification_indices
        ), "Invalid modification index."
        # Construct the peptide string
        if len(residue_indices) == 0:
            return ""
        if len(residue_indices) != len(modification_indices):
            raise ValueError(
                "Residue and modification indices must have the same length."
            )
        if any(index == self.residue_pad_index for index in residue_indices):
            raise ValueError("Residue indices contain padding index.")
        if any(index == self.modification_pad_index for index in modification_indices):
            raise ValueError("Modification indices contain padding index.")
        peptide = "".join(
            self.residue_vocabulary[residue_index]
            + self.modifications_vocabulary[modification_index]
            for residue_index, modification_index in zip(
                residue_indices, modification_indices
            )
        )
        return peptide

    def pad_tokenize(
        self, peptides: list[str], max_length: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Tokenize a list of modified peptides and pad them to the same length.

        Args:
            peptides: List of modified peptide strings (e.g., ['ACDEFGHIK[UNIMOD:35]', 'LMNPQ']).

        Returns:
            A tuple of two jnp.ndarrays: padded residue indices and padded modification indices,
            along with a padding mask indicating valid positions.
        """
        if peptides is None:
            return jnp.array([]), jnp.array([]), jnp.array([])
        if not all(isinstance(peptide, str) for peptide in peptides):
            raise ValueError("All elements in the peptides list must be strings.")
        if any(not peptide for peptide in peptides):
            raise ValueError("Peptides list contains empty strings.")
        if any(not re.match(r"^[A-Z\[\]0-9:]+$", peptide) for peptide in peptides):
            raise ValueError(
                "Peptides list contains invalid characters. Only uppercase letters, brackets, and digits are allowed."
            )

        residues_array, modifications_array, lengths = [], [], []

        for peptide in peptides:
            residue_indices, modification_indices = self.tokenize(peptide=peptide)
            residues_array.append(jnp.array(residue_indices, dtype=int))
            modifications_array.append(jnp.array(modification_indices, dtype=int))
            if len(residue_indices) >= max_length:
                print(f"Peptide {peptide} is longer than {max_length} with residue indices {residue_indices}.")
            lengths.append(len(residue_indices))

        residues_array = jnp.stack(
            [
                jnp.pad(
                    residue_array,
                    (0, max_length - length),
                    constant_values=self.residue_pad_index,
                )
                for residue_array, length in zip(residues_array, lengths)
            ]
        )
        modifications_array = jnp.stack(
            [
                jnp.pad(
                    modification_array,
                    (0, max_length - length),
                    constant_values=self.modification_pad_index,
                )
                for modification_array, length in zip(modifications_array, lengths)
            ]
        )
        padding_mask = jnp.arange(0, max_length).reshape(1, -1) < jnp.array(
            lengths
        ).reshape(-1, 1)
        return residues_array, modifications_array, padding_mask
