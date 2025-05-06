import { useEffect, useState } from "react";
import Papa from "papaparse";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';
import { ChartDataset, ChartData } from 'chart.js';

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
  const [data, setData] = useState<ChartData<'line', number[]> | null>(null);
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

        const datasets: ChartDataset<'line', number[]>[] = [{
          label: "净值曲线",
          data: values,
          fill: false,
          tension: 0.3,
          borderColor: '#4f46e5', // 添加颜色配置
        }];

        setData({
          labels: dates,
          datasets
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

      {/* 指标卡片展示 */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", marginBottom: "2rem" }}>
        {[
          { label: "总收益率", value: "+124.53%" },
          { label: "年化收益率", value: "42.78%" },
          { label: "总交易次数", value: "87" },
          { label: "盈利因子", value: "1.85" },
          { label: "最大回撤", value: "-18.42%" },
          { label: "夏普比率", value: "2.14" },
        ].map((item, idx) => (
          <div key={idx} style={{
            flex: "1 0 30%",
            padding: "1rem",
            border: "1px solid #ddd",
            borderRadius: "8px",
            backgroundColor: "#f9fafb",
            minWidth: "140px",
            textAlign: "center"
          }}>
            <div style={{ fontSize: "14px", color: "#666" }}>{item.label}</div>
            <div style={{ fontSize: "20px", fontWeight: "bold", marginTop: "0.5rem" }}>{item.value}</div>
          </div>
        ))}
      </div>

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