from dataclasses import dataclass
import logging
import sys
from pathlib import Path


@dataclass
class StreamToLogger:
    file_path: str | None = None
    log_level: int = logging.INFO

    def __post_init__(self):
        self.original_stdout = sys.stdout
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler for logging to file
        self.file_handler = None
        if self.file_path is not None:
            self.add_file_handler(self.file_path)

        # Stream handler for logging to console
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter('%(message)s')
        self.console_handler.setFormatter(console_formatter)
        self.logger.addHandler(self.console_handler)

        self.start()

    def add_file_handler(self, file_path: str):
        self.file_path = file_path
        if not Path(self.file_path).exists():
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        self.file_handler = logging.FileHandler(file_path)
        self.file_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        if self.file_handler:
            self.file_handler.flush()
        self.console_handler.flush()

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.original_stdout
        if self.file_handler:
            self.logger.info(f'Logs saved to {self.file_path}')
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)