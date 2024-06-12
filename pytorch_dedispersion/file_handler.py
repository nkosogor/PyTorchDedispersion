import os
from urllib.request import urlretrieve

class FileHandler:
    def __init__(self, source, download_dir=None):
        self.source = source
        self.download_dir = download_dir

    def load_file(self):
        if self.is_url(self.source):
            return self.download_file(self.source)
        elif os.path.exists(self.source):
            return self.source
        else:
            raise ValueError("Invalid source: Must be a valid URL or file path")

    def is_url(self, source):
        return source.startswith('http://') or source.startswith('https://')

    def download_file(self, url):
        if not self.download_dir:
            raise ValueError("Download directory not specified for URL source")
        filename = os.path.join(self.download_dir, os.path.basename(url))
        urlretrieve(url, filename)
        return filename
