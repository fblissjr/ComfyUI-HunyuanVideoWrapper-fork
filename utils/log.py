import logging
import os

def setup_logger(name, log_file='comfy_hyvideo.log', level=logging.INFO):
    """
    Sets up a logger that outputs to both a file and the console.
    """
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # file output
    log_file_path = os.path.join(log_directory, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

# Example usage in other modules:
# from ..utils.log import setup_logger
# log = setup_logger(__name__)
# log.info("This will be logged to console and file!")