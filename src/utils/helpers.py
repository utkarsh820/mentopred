import os
import logging
from datetime import datetime
import traceback

def setup_logging(log_dir="logs"):
    """Initializes logging with timestamped file and console output."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Console logging for visibility during development
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

class CustomException(Exception):
    """Custom exception with traceback and logging."""
    def __init__(self, errormessage):
        super().__init__(errormessage)
        self.errormessage = errormessage
        self.trace = traceback.format_exc()

        # Log the error with traceback
        logging.error(f"{self.trace.strip()} | {self.errormessage}")

    def __str__(self):
        return f"{self.errormessage}\nTraceback:\n{self.trace}"