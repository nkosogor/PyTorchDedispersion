from typing import NoReturn
import os

class FileHandler:
    def __init__(self, source: str) -> None:
        """
        Initialize FileHandler.

        Args:
            source (str): Path to the data file.
        """
        self.source = source

    def load_file(self) -> str:
        """
        Load the data file.

        Returns:
            str: Path to the data file if it exists.
        """
        if os.path.exists(self.source):
            return self.source
        else:
            raise ValueError("Invalid source: Must be a valid file path")
