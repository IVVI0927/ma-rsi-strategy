import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

interface ScoreItem {
    code: string;
    score: number;
  }

export default function ScoreDashboard() {
  const [todayScores, setTodayScores] = useState<ScoreItem[]>([]);
  const [cachedScores, setCachedScores] = useState<Record<string, number>>({});
  const [esScores, setEsScores] = useState<ScoreItem[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/get_today_scores")
      .then(res => res.json())
      .then(data => setTodayScores(data));

    fetch("http://localhost:8000/api/get_cached_scores")
      .then(res => res.json())
      .then(data => setCachedScores(data));

    fetch("http://localhost:8000/api/search_scores?min_score=0.85")
      .then(res => res.json())
      .then(data => setEsScores(data));
  }, []);

  return (
    <div className="p-4 space-y-6">
      <section>
        <h2 className="text-xl font-bold mb-2">🎯 今日推荐（CSV）</h2>
        <ul className="list-disc ml-5">
          {todayScores.map((item, idx) => (
            <li key={idx}>
              <Link to={`/stock/${item.code}`} className="text-blue-600 hover:underline">
                {item.code} - 得分: {item.score}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-xl font-bold mb-2">📦 Redis 缓存评分</h2>
        <ul className="list-disc ml-5">
          {Object.entries(cachedScores).map(([code, score]) => (
            <li key={code}>{code} - 缓存得分: {score}</li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-xl font-bold mb-2">🔍 Elasticsearch 高分推荐</h2>
        <ul className="list-disc ml-5">
          {esScores.map((item, idx) => (
            <li key={idx}>
              <Link to={`/stock/${item.code}`} className="text-blue-600 hover:underline">
                {item.code} - 得分: {item.score}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}