from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from signal_engine.score_and_suggest import get_today_scores
from redis_score_store import RedisScoreStore
from es.es_client import es

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/get_today_scores")
def fetch_today_scores():
    return get_today_scores()

@app.get("/api/get_cached_scores")
def get_cached_scores():
    """从 Redis 获取今日评分缓存"""
    return RedisScoreStore().load_score()

@app.get("/api/search_scores")
def search_scores(min_score: float = 0.8):
    """从 Elasticsearch 中搜索得分高于阈值的记录"""
    result = es.search(index="stock_scores", body={
        "query": {
            "range": {
                "score": {
                    "gte": min_score
                }
            }
        }
    })
    return [hit["_source"] for hit in result["hits"]["hits"]]
