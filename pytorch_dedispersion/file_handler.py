import os

class FileHandler:
    def __init__(self, source):
        self.source = source

    def load_file(self):
        if os.path.exists(self.source):
            return self.source
        else:
            raise ValueError("Invalid source: Must be a valid file path")
