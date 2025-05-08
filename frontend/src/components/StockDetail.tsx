import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';

export default function StockDetail() {
  const { code } = useParams();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch(`http://localhost:8000/api/score_detail?code=${code}`)
      .then((res) => res.json())
      .then((res) => setData(res));
  }, [code]);

  if (!data) return <div className="p-4">加载中...</div>;

  return (
    <div className="p-4 space-y-2">
      <h2 className="text-xl font-bold">股票代码：{data.code}</h2>
      <p>综合得分：{data.score}</p>
      <p>推荐理由：{data.reason}</p>
      <hr />
      <div>
        <h3 className="font-semibold">📊 因子详情：</h3>
        <ul className="list-disc ml-5">
          {Object.entries(data).map(([key, value]: [string, any]) => {
            if (["code", "score", "reason"].includes(key)) return null;
            return (
              <li key={key}>
                {key}: {value}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}