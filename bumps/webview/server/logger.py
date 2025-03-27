import logging

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a logger and allow all messages through. We will use separate handlers
# to control the level going to the server console and to the client.
logger = logging.getLogger("webview")
logger.setLevel(logging.DEBUG)

# Prepare for capture_warnings, in case we want to turn warnings.warn()
# notifications into logging items.
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.setLevel(logging.WARNING)


def setup_console_logging(level):
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.WARNING)
    handler.setLevel(level)
    logger.addHandler(handler)
    warnings_logger.addHandler(handler)
    # logger.error("Are errors printed?")
    # logger.warning("Are warnings printed?")
    # logger.info("Is info printed?")
    # logger.debug("Are debug messages printed?")
    # import warnings; warnings.warn("Test warning")
    return handler


# TODO: replace emit with something that communicates with the client
class ClientHandler(logging.Handler):
    def __init__(self, action=None):
        super().__init__()
        self.action = action
        self.log_list = []

    def emit(self, record):
        # print("logging", record)
        self.log_list.append(self.format(record))
        if self.action is not None:
            self.action(self.log_list)


def setup_client_logging(level, action=None):
    handler = ClientHandler(action)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler


# TODO: do we want warnings from other packages going to the logger?
def monkeypatch_warnings_formatter(strip_newline=False):
    """
    Modify the global the warnings formatter to get rid of the source code display.
    If *strip_newline* then remove the trailing linefeed character as well.

    Use warnings.formatwarning.restore() to recover original warnings formatting.
    """
    import warnings

    _formatwarning = warnings.formatwarning

    def exclude_source_line(message, category, filename, lineno, line=None):
        """Drop the source code for the line and maybe the trailing linefeed"""
        message = _formatwarning(message, category, filename, lineno, line="")
        return message.strip() if strip_newline else message

    def restore_warnings_formatter():
        warnings.formatwarning = _formatwarning

    exclude_source_line.restore = restore_warnings_formatter
    warnings.formatwarning = exclude_source_line


def capture_warnings(monkeypatch=False):
    # import warnings; warnings.warn("test warnings before")
    if monkeypatch:
        monkeypatch_warnings_formatter(strip_newline=True)
    logging.captureWarnings(True)
    # import warnings; warnings.warn("test warnings after")


# # Verifying that the level on the handler can be different from the level on the logger
# test_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s [TEST]")
# test_handler = logging.StreamHandler()
# test_handler.setFormatter(test_formatter)
# logger.addHandler(test_handler)
