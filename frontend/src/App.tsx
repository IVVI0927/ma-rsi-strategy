import BacktestDashboard from "./pages/BacktestDashboard";
import ScoreDashboard from './components/ScoreDashboard';
import StockDetail from './components/StockDetail';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Link } from 'react-router-dom';


function App() {
  try {
    return (
      <BrowserRouter>
        <div className="p-4 space-y-4">
          <nav className="flex space-x-4 mb-4">
            <Link to="/" className="text-blue-600 hover:underline">评分首页</Link>
            <Link to="/backtest" className="text-blue-600 hover:underline">回测数据</Link>
          </nav>
          <Routes>
            <Route path="/" element={<ScoreDashboard />} />
            <Route path="/backtest" element={<BacktestDashboard />} />
            <Route path="/stock/:code" element={<StockDetail />} />
          </Routes>
        </div>
      </BrowserRouter>
    );
  } catch (e) {
    return <div>❌ 页面渲染出错，请检查控制台日志。</div>;
  }
}

export default App;