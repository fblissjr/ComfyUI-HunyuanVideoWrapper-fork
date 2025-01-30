import logging
import pytest
from utils.log import setup_logger

def test_logging_setup():
    # Test with standard logging
    log = setup_logger("test_logger_std", log_file="test_std.log")

    assert isinstance(
        log, logging.Logger
    ), "Standard logger should be a logging.Logger instance"

    # Verify stream handler is configured
    assert any(
        isinstance(h, logging.StreamHandler) for h in log.handlers
    ), "Stream handler not configured"

    # Verify file handler is configured
    assert any(
        isinstance(h, logging.FileHandler) for h in log.handlers
    ), "File handler not configured"

    # Basic test to ensure logging doesn't throw an error
    log.info("This is a standard log message.")

def test_structlog_setup():
    # Test with structlog
    log = setup_logger("test_logger_struct", log_file="test_struct.log", enable_structlog=True)

    assert isinstance(
        log, structlog.stdlib.BoundLogger
    ), "Structlog logger should be a structlog.stdlib.BoundLogger instance"

    # Verify stream handler is configured
    logger = logging.getLogger("test_logger_struct")  # Get underlying logger
    assert any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ), "Stream handler not configured"

    # Verify file handler is configured
    assert any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ), "File handler not configured"

    # Basic test to ensure logging doesn't throw an error
    log.info("This is a structlog message.", event="test_event")
    
    
# # Using standard logging (in any module)
# from ..utils.log import setup_logger
# log = setup_logger(__name__)

# log.debug("This is a debug message.")
# log.info("This is an info message.")
# log.warning("This is a warning.")
# log.error("This is an error!")
# log.critical("This is a critical error!!!")

# # Using structlog for debugging/observability (in specific modules)
# from ..utils.log import setup_logger
# log = setup_logger(__name__, enable_structlog=True)

# log.debug("Debugging event.", event="debug", variable_x=123, some_data=[1,2,3])
# log.info("User logged in.", event="user_login", user_id=42, ip_address="192.168.1.1")
# log.warning("Suspicious activity.", event="security", reason="unusual_access_pattern")