import logging
import os
import sys


def get_logging_writer(writer_name: str, recording_directory:str) -> logging.Logger:
    """
    Creates out and error writers for the current logger. When called, the writers will print whatever is being logged
    to the console and either a "out.log" or "err.log" file in the directory of the current trial
    Args:
        writer_name: The name of the writer. Will be used as a prefix for everything written by it
        recording_directory: Directory to record to

    Returns: The writer (or rather a native python logger) that can be used to log things

    """
    custom_formatter = CustomFormatter()

    writer = logging.getLogger(name=writer_name)
    writer.setLevel(logging.INFO)

    # add stream to terminal
    out_stream_handler = logging.StreamHandler(sys.stdout)
    out_stream_handler.setLevel(logging.INFO)  # log everything but DEBUG logs
    out_stream_handler.setFormatter(custom_formatter)
    writer.addHandler(out_stream_handler)

    error_stream_handler = logging.StreamHandler()  # uses stderr by default
    error_stream_handler.setLevel(logging.ERROR)
    error_stream_handler.setFormatter(custom_formatter)
    writer.addHandler(error_stream_handler)

    # add stream to recording file

    file_handler = logging.FileHandler(os.path.join(recording_directory, "out.log"))
    file_handler.setFormatter(custom_formatter)
    file_handler.setLevel(logging.INFO)
    writer.addHandler(file_handler)

    error_file_handler = logging.FileHandler(os.path.join(recording_directory, 'err.log'))
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(custom_formatter)
    writer.addHandler(error_file_handler)
    return writer


class CustomFormatter(logging.Formatter):
    """
    Custom Formatter to allow for uniform and pretty logging
    """

    def __init__(self):
        super().__init__()
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s]:[%(name)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)  # everything but CRITICAL is formatted via standard
        else:
            return self.red_formatter.format(record)  # CRITICAL logs are formatted via red