import logging
import os
import structlog

def setup_logger(name, log_file='comfy_hyvideo.log', level=logging.INFO, enable_structlog=False):
    """
    Sets up a logger that can use either structlog or the standard logging library.

    Args:
        name (str): The name of the logger.
        log_file (str): The name of the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        enable_structlog (bool): Whether to enable structlog or use standard logging.
    """

    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    if enable_structlog:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Wrap stdlib with extra attributes
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Create logger instance
        logger = structlog.get_logger(name)

        # Set up formatter for structlog
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(), # You can change this as needed
        )
    else:
        # Configure standard logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logger instance
        logger = logging.getLogger(name)
        logger.setLevel(level)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    log_file_path = os.path.join(log_directory, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger