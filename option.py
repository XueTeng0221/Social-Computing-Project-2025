import os
import logging

os.environ["HF-ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w')
logger = logging.getLogger(__name__)