import logging

logger = logging.getLogger("model_server")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("../logs/model_server.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
