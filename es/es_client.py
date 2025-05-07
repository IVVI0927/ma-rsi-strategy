from elasticsearch import Elasticsearch

# 创建连接本地的 Elasticsearch 服务
es = Elasticsearch("http://localhost:9200")