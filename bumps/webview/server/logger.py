import logging

class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_list = []

    def emit(self, record):
        self.log_list.append(self.format(record))

# Create a logger
logger = logging.getLogger('webview')
logger.setLevel(logging.INFO)

# Create and add the custom handler
list_handler = ListHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
list_handler.setFormatter(formatter)
list_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)