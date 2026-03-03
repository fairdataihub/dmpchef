import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    """
    A reusable logger that creates both console and file logs,
    with structured JSON formatting for easy analysis.
    """

    def __init__(self, log_dir="logs"):
        # Create logs directory if it doesn't exist
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create timestamped log file
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        """
        Returns a configured structlog logger for the given module name.
        """
        logger_name = os.path.basename(name)

        # --- File Handler ---
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # --- Console Handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # --- Configure base logging ---
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[console_handler, file_handler],
            force=True  # ensures reconfiguration even if logging already initialized
        )

        # --- Configure structlog ---
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)


#  Create a single global logger to use everywhere
GLOBAL_LOGGER = CustomLogger().get_logger(__name__)



