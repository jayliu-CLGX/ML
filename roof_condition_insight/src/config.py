import os
import logging

# Base URLs for authentication and roof condition insights API
BASE_URLS_AUTH = {
    "UAT": "https://uatauth.myriadexchange.com",
    "PRD": "https://auth.myriadexchange.com"
}
BASE_URLS_RCI = {
    "UAT": "https://uatdigitalhubapi.myriadexchange.com",
    "PRD": "https://digitalhubapi.myriadexchange.com"
}

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PDF_DIR = os.path.join(DATA_DIR, 'pdfs')
JSON_DIR = os.path.join(DATA_DIR, 'jsons')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Logging configuration
LOGGING_CONFIG = {
    'filename': os.path.join(LOGS_DIR, 'roof_condition_insights.log'),
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}
