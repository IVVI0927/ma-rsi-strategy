from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from signal_engine.score_and_suggest import get_today_scores

app = FastAPI()

# ✅ 跨域设置（确保前端可访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定前端地址如 "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 定义 API 路由
@app.get("/api/get_today_scores")
def fetch_today_scores():
    return get_today_scores()

# ✅ 如果你有前端打包好的dist目录，可以这样挂载静态文件
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")