"""Configuration output formatter with hierarchical colour-coding."""

from rich.console import Console
from rich.text import Text
from omegaconf import DictConfig, OmegaConf


class ConfigFormatter:
    """Format Hydra configuration with hierarchical colour-coding based on nesting depth.

    Keys are coloured according to their indentation level to help visualise
    the configuration structure.
    """

    # Colour palette for different indentation levels (similar to Typer's style)
    INDENT_COLOURS = [
        "bright_cyan",  # Level 0 (root keys)
        "bright_green",  # Level 1
        "bright_yellow",  # Level 2
        "bright_magenta",  # Level 3
        "bright_blue",  # Level 4
        "cyan",  # Level 5
        "green",  # Level 6
        "yellow",  # Level 7+
    ]

    def __init__(self):
        """Initialise the formatter."""
        self.console = Console()

    def print_config(self, cfg: DictConfig) -> None:
        """Print configuration with hierarchical colour-coding.

        Args:
            cfg: OmegaConf configuration object to format and print
        """
        yaml_str = OmegaConf.to_yaml(cfg)
        output = Text()

        for line in yaml_str.split("\n"):
            formatted_line = self._format_line(line)
            output.append(formatted_line)

        self.console.print(output, end="")

    def _format_line(self, line: str) -> Text:
        """Format a single line of YAML with appropriate colouring.

        Args:
            line: A single line from the YAML output

        Returns:
            Rich Text object with formatted content
        """
        output = Text()

        # Handle empty lines
        if not line.strip():
            output.append("\n")
            return output

        indent_level = self._get_indent_level(line)
        colour = self._get_colour_for_level(indent_level)

        # Handle list items specially (they contain '- ' prefix)
        if self._is_list_item(line):
            output.append(line)
            output.append("\n")
            return output

        # Handle key-value pairs
        separator_idx = self._find_key_value_separator(line)
        if separator_idx != -1:
            self._append_key_value_pair(output, line, separator_idx, colour)
        else:
            # Lines without key-value separator
            output.append(line)
            output.append("\n")

        return output

    def _get_indent_level(self, line: str) -> int:
        """Calculate the indentation level of a line.

        Args:
            line: Line to analyse

        Returns:
            Indentation level (0 for root, 1 for first nested level, etc.)
        """
        return (len(line) - len(line.lstrip())) // 2

    def _get_colour_for_level(self, indent_level: int) -> str:
        """Get the colour for a given indentation level.

        Args:
            indent_level: The indentation level

        Returns:
            Colour name for Rich
        """
        return self.INDENT_COLOURS[min(indent_level, len(self.INDENT_COLOURS) - 1)]

    def _is_list_item(self, line: str) -> bool:
        """Check if a line is a YAML list item.

        Args:
            line: Line to check

        Returns:
            True if line is a list item (starts with '- ')
        """
        return line.lstrip().startswith("- ")

    def _find_key_value_separator(self, line: str) -> int:
        """Find the position of the YAML key-value separator.

        This finds colons that are followed by a space or end of line,
        avoiding colons inside keys like M[UNIMOD:35].

        Args:
            line: Line to search

        Returns:
            Index of the separator colon, or -1 if not found
        """
        for i, char in enumerate(line):
            if char == ":":
                # Check if this is followed by space, end of line, or is the last char
                if i + 1 >= len(line) or line[i + 1] == " ":
                    return i
        return -1

    def _append_key_value_pair(
        self, output: Text, line: str, separator_idx: int, colour: str
    ) -> None:
        """Append a formatted key-value pair to the output.

        Args:
            output: Text object to append to
            line: Original line
            separator_idx: Index of the separator colon
            colour: Colour to use for the key
        """
        key_part = line[:separator_idx]
        value_part = line[separator_idx + 1 :]
        indent = " " * (len(line) - len(line.lstrip()))

        # Add indentation
        output.append(indent)

        # Add coloured key
        output.append(key_part.lstrip(), style=f"bold {colour}")
        output.append(":")

        # Add value without formatting (plain text)
        if value_part:
            output.append(value_part)

        output.append("\n")
