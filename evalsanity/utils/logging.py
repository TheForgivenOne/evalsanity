import sys
import datetime


class Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg, level="INFO"):
        if self.verbose:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}", file=sys.stderr, flush=True)

    def info(self, msg):
        self.log(msg, "INFO")

    def warn(self, msg):
        self.log(msg, "WARN")

    def error(self, msg):
        self.log(msg, "ERROR")

    def debug(self, msg):
        self.log(msg, "DEBUG")
