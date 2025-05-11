from elasticsearch import Elasticsearch
import logging

try:
    es = Elasticsearch(
        "http://localhost:9200",  # 默认本地ES地址
        basic_auth=("elastic", "your_password"),  # 如果有认证
        verify_certs=False  # 开发环境可以关闭证书验证
    )
    
    # 测试连接
    if not es.ping():
        logging.warning("Elasticsearch connection failed")
        es = None
except Exception as e:
    logging.error(f"Elasticsearch connection error: {e}")
    es = None