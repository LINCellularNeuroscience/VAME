import logging
from pathlib import Path
import io


class VameLogger:
    LOG_FORMAT = (
        "%(asctime)-15s.%(msecs)d %(levelname)-5s --- [%(threadName)s]"
        " %(name)-15s : %(lineno)d : %(message)s"
    )
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, base_name: str, file_path: str | None = None, log_level: int = logging.INFO):
        self.log_level = log_level
        self.file_handler = None
        logging.basicConfig(
            level=log_level,
            format=self.LOG_FORMAT,
            datefmt=self.LOG_DATE_FORMAT
        )
        self.logger = logging.getLogger(f'{base_name}')
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(self.log_level)
        # Stream handler for logging to stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(self.LOG_FORMAT, self.LOG_DATE_FORMAT)
        )
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False

        if file_path is not None:
            self.add_file_handler(file_path)

    def add_file_handler(self, file_path: str):
        # File handler for logging to a file
        if not Path(file_path).parent.exists():
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        if self.file_handler is not None:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)

        self.file_handler = logging.FileHandler(file_path, mode='w')
        self.file_handler.setFormatter(logging.Formatter(self.LOG_FORMAT, self.LOG_DATE_FORMAT))
        self.logger.addHandler(self.file_handler)

    def remove_file_handler(self):
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None



class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)