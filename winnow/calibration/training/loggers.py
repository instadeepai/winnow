import dataclasses
import logging
from typing import List, Protocol

import neptune
from rich.logging import RichHandler


class Logger(Protocol):
    """Logger that logs to a specific platform."""

    def log_metric(self, name: str, metric: int | float, step: int) -> None:
        """Log a metric to the platform.

        Args:
            name: The name of the metric.
            metric: The metric value.
            step: The step number.
        """
        pass

    def log_message(self, message: str) -> None:
        """Log a message to the platform.

        Args:
            message: The message to log.
        """
        pass


@dataclasses.dataclass
class LoggingManager:
    """Manager that logs to multiple platforms."""

    loggers: List[Logger]

    def log_metric(self, name: str, metric: int | float, step: int) -> None:
        """Log a metric to all platforms.

        Args:
            name: The name of the metric.
            metric: The metric value.
            step: The step number.
        """
        for logger in self.loggers:
            logger.log_metric(name=name, metric=metric, step=step)

    def log_message(self, message: str) -> None:
        """Log a message to all platforms.

        Args:
            message: The message to log.
        """
        for logger in self.loggers:
            logger.log_message(message=message)


class NeptuneLogger(Logger):
    """Logger that logs to Neptune."""

    def __init__(self, project_name: str, api_token: str):
        self.project_name = project_name
        self.neptune_logger = neptune.init_run(
            project=project_name, api_token=api_token
        )

    def __str__(self) -> str:
        return f"NeptuneLogger(project_name={self.project_name})"

    def log_metric(self, name: str, metric: int | float, step: int) -> None:
        """Log a metric to Neptune.

        Args:
            name: The name of the metric.
            metric: The metric value.
            step: The step number.
        """
        self.neptune_logger[name].append(value=metric, step=step)

    def __repr__(self) -> str:
        return self.__str__()


class SystemLogger(Logger):
    """Logger that logs to a file and the console."""

    def __init__(self, log_file: str):
        # Create a logger
        self.log_file = log_file
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        # Create a stream handler
        self.stream_handler = RichHandler()
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def __str__(self) -> str:
        return f"SystemLogger(log_file={self.log_file})"

    def log_metric(self, name: str, metric: int | float, step: int) -> None:
        """Log a metric to the file and the console.

        Args:
            name: The name of the metric.
            metric: The metric value.
            step: The step number.
        """
        self.logger.info(f"{name}: {metric} at step {step}")

    def log_message(self, message: str) -> None:
        """Log a message to the file and the console.

        Args:
            message: The message to log.
        """
        self.logger.info(message)

    def __repr__(self) -> str:
        return self.__str__()
