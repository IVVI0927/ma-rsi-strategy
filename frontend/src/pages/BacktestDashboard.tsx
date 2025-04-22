import { useEffect, useState } from "react";
import Papa from "papaparse";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';

// 注册Chart.js必要组件
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement);

type BacktestRow = {
  date: string;
  cash: string;
  market_value: string;
  total_asset: string;
  holdings: string;
};

export default function BacktestDashboard() {
  const [data, setData] = useState<{
    labels: string[];
    datasets: { label: string; data: number[]; fill: boolean; tension: number }[];
  } | null>(null);
  const [rows, setRows] = useState<BacktestRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsLoading(true);
    fetch("/backtest_result.csv")
      .then((res) => {
        if (!res.ok) throw new Error("HTTP状态码异常");
        return res.text();
      })
      .then((text) => {
        const parsed = Papa.parse<BacktestRow>(text, {
          header: true,
          skipEmptyLines: true,
          transformHeader: h => h.trim(),
          transform: (value) => value.trim(),
        });

        if (parsed.errors.length > 0) {
          throw new Error("CSV解析错误");
        }

        const cleaned = parsed.data.filter((r) => r.date);
        if (cleaned.length === 0) {
          throw new Error("无有效数据");
        }

        const dates = cleaned.map((r) => r.date);
        const values = cleaned.map((r) => parseFloat(r.total_asset));

        setData({
          labels: dates,
          datasets: [{
            label: "净值曲线",
            data: values,
            fill: false,
            tension: 0.3,
            borderColor: '#4f46e5', // 添加颜色配置
          }],
        });
        setRows(cleaned);
      })
      .catch((err) => {
        setError(err.message || "数据加载失败");
      })
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ fontSize: "24px", fontWeight: "bold", marginBottom: "1rem" }}>📈 策略净值曲线</h2>
      
      {isLoading && <div>加载中...</div>}
      
      {error && (
        <div style={{ color: "red", marginBottom: "1rem" }}>
          ❌ 错误: {error}
        </div>
      )}

      {data && !error && (
        <Line 
          data={data} 
          options={{
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: false } }
          }}
        />
      )}

      <h2 style={{ fontSize: "20px", fontWeight: "600", marginTop: "2rem" }}>📋 回测每日明细</h2>
      
      {rows.length > 0 ? (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
          {/* ...原有表格代码... */}
        </table>
      ) : (
        !isLoading && <div>暂无数据</div>
      )}
    </div>
  );
}