from dataclasses import dataclass
import logging
import sys


@dataclass
class StreamToLogger:
    filename: str
    log_level: int = logging.INFO
    linebuf: str = ''

    def __post_init__(self):
        self.original_stdout = sys.stdout
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)


        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler for logging to file
        self.file_handler = logging.FileHandler(self.filename)
        self.file_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        # Stream handler for logging to console
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter('%(message)s')
        self.console_handler.setFormatter(console_formatter)
        self.logger.addHandler(self.console_handler)

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        self.file_handler.flush()

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.original_stdout
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)