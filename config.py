import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
JQDATA_USER = os.getenv("JQDATA_USER")
JQDATA_PASS = os.getenv("JQDATA_PASS") 