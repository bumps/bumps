import logging

LOGLEVEL = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warn=logging.WARNING,
    warning=logging.WARNING,
    error=logging.ERROR,
    critical=logging.CRITICAL,
)

_CONSOLE_FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_CONSOLE_HANDLER = None

# Create a logger and allow all messages through. We will use separate handlers
# to control the level going to the server console and to the client.
logger = logging.getLogger("webview")
logger.setLevel(logging.DEBUG)

# Prepare for capture_warnings, in case we want to turn warnings.warn()
# notifications into logging items.
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.setLevel(logging.WARNING)


def setup_console_logging(level):
    global _CONSOLE_HANDLER
    if _CONSOLE_HANDLER is None:
        _CONSOLE_HANDLER = logging.StreamHandler()
        _CONSOLE_HANDLER.setFormatter(_CONSOLE_FORMATTER)
        logger.addHandler(_CONSOLE_HANDLER)
        warnings_logger.addHandler(_CONSOLE_HANDLER)
        # logger.error("Are errors printed?")
        # logger.warning("Are warnings printed?")
        # logger.info("Is info printed?")
        # logger.debug("Are debug messages printed?")
        # import warnings; warnings.warn("Test warning")
    _CONSOLE_HANDLER.setLevel(LOGLEVEL[level])
    return _CONSOLE_HANDLER


# From PHIND-70B-Model
def print_logging_configuration():
    # Print root logger configuration
    print("\nRoot Logger Configuration:")
    root = logging.getLogger()
    print(f"Root logger level: {logging.getLevelName(root.level)}")

    # Print all handlers and their configurations
    print("\nHandlers:")
    for handler in root.handlers:
        print(f"- Handler type: {handler.__class__.__name__}")
        print(f"  Level: {logging.getLevelName(handler.level)}")

        # Print formatter details if available
        if hasattr(handler, "formatter"):
            print(f"  Formatter format: {handler.formatter._fmt}")

    # Print all logger configurations
    print("\nAll Loggers:")
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if len(logger.handlers) > 0:
            print(f"\nLogger '{logger.name}':")
            print(f"Level: {logging.getLevelName(logger.level)}")
            for handler in logger.handlers:
                print(f"- Handler type: {handler.__class__.__name__}")
                print(f"  Level: {logging.getLevelName(handler.level)}")


# TODO: unused code: ClientHandler, setup_client_logging
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
    handler.setFormatter(_CONSOLE_FORMATTER)
    handler.setLevel(LOGLEVEL[level])
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
