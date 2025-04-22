import BacktestDashboard from "./pages/BacktestDashboard";

function App() {
  try {
    return <BacktestDashboard />;
  } catch (e) {
    return <div>❌ 页面渲染出错，请检查控制台日志。</div>;
  }
}

export default App;