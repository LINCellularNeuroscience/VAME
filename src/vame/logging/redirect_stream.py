from dataclasses import dataclass
import logging


@dataclass
class StreamToLogger:
    filename: str
    log_level: int = logging.INFO
    linebuf: str = ''

    def __post_init__(self):
        logging.basicConfig(filename=self.filename, level=self.log_level, format='%(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(self.log_level)

        self.file_handler = logging.FileHandler(self.filename)
        self.file_handler.setLevel(self.log_level)

        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        self.file_handler.flush()