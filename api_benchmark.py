#!/usr/bin/env python3
"""
API Performance Benchmark for A-Share Quantitative Trading System
Tests FastAPI endpoints with varying concurrent loads
"""

import asyncio
import time
import statistics
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import httpx
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIBenchmark:
    """API performance benchmark runner"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    async def single_request(self, client: httpx.AsyncClient, endpoint: str, method: str = "GET", json_data: dict = None) -> Dict[str, Any]:
        """Make a single API request and measure performance"""
        start_time = time.perf_counter()
        
        try:
            if method == "GET":
                response = await client.get(f"{self.base_url}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{self.base_url}{endpoint}", json=json_data)
            
            end_time = time.perf_counter()
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'content_length': len(response.content) if response.content else 0,
                'error': None
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'status_code': 0,
                'response_time': end_time - start_time,
                'content_length': 0,
                'error': str(e)
            }
    
    async def concurrent_requests(self, endpoint: str, concurrent_users: int, 
                                method: str = "GET", json_data: dict = None, 
                                duration: int = 30) -> List[Dict[str, Any]]:
        """Run concurrent requests against an endpoint"""
        logger.info(f"Testing {endpoint} with {concurrent_users} concurrent users for {duration}s...")
        
        results = []
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - start_time < duration:
                # Create concurrent tasks
                tasks = []
                for _ in range(concurrent_users):
                    task = self.single_request(client, endpoint, method, json_data)
                    tasks.append(task)
                
                # Execute concurrent requests
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, dict):
                        results.append(result)
                    else:
                        # Handle exceptions
                        results.append({
                            'success': False,
                            'status_code': 0,
                            'response_time': 0,
                            'content_length': 0,
                            'error': str(result)
                        })
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze API performance results"""
        if not results:
            return {}
        
        # Filter successful requests
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'total_requests': len(results),
                'successful_requests': 0,
                'error_rate': 1.0,
                'response_times': {},
                'throughput': 0
            }
        
        response_times = [r['response_time'] for r in successful_results]
        
        # Calculate percentiles
        percentiles = {
            'p50': np.percentile(response_times, 50),
            'p90': np.percentile(response_times, 90), 
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99),
            'min': min(response_times),
            'max': max(response_times),
            'mean': statistics.mean(response_times),
            'median': statistics.median(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
        
        # Calculate throughput (requests per second)
        total_time = sum(response_times)
        throughput = len(successful_results) / total_time if total_time > 0 else 0
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(results) - len(successful_results),
            'error_rate': (len(results) - len(successful_results)) / len(results),
            'response_times': percentiles,
            'throughput': throughput,
            'avg_content_length': statistics.mean([r['content_length'] for r in successful_results]) if successful_results else 0
        }
    
    async def benchmark_endpoint(self, endpoint: str, test_name: str, 
                                concurrent_loads: List[int] = [1, 10, 50, 100], 
                                method: str = "GET", json_data: dict = None,
                                duration: int = 10) -> Dict[str, Any]:
        """Benchmark an endpoint with different concurrent loads"""
        logger.info(f"Benchmarking {test_name} ({endpoint})...")
        
        endpoint_results = {}
        
        for load in concurrent_loads:
            results = await self.concurrent_requests(endpoint, load, method, json_data, duration)
            analysis = self.analyze_results(results)
            endpoint_results[f"{load}_users"] = analysis
            
            logger.info(f"  {load} users: P95={analysis['response_times'].get('p95', 0)*1000:.1f}ms, "
                       f"Error rate={analysis.get('error_rate', 0)*100:.1f}%")
        
        return endpoint_results
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive API benchmarks"""
        logger.info("Starting comprehensive API benchmark...")
        
        # Test configurations
        test_configs = [
            {
                'endpoint': '/',
                'test_name': 'Root Endpoint',
                'method': 'GET',
                'concurrent_loads': [1, 10, 50, 100, 250]
            },
            {
                'endpoint': '/recommend',
                'test_name': 'Stock Recommendations',
                'method': 'GET', 
                'concurrent_loads': [1, 10, 25, 50]  # Lower loads for compute-heavy endpoint
            },
            {
                'endpoint': '/recommend_portfolio/',
                'test_name': 'Portfolio Builder',
                'method': 'POST',
                'json_data': {'capital': 100000, 'max_stocks': 5},
                'concurrent_loads': [1, 5, 10, 20]
            }
        ]
        
        benchmark_results = {}
        
        for config in test_configs:
            try:
                results = await self.benchmark_endpoint(
                    endpoint=config['endpoint'],
                    test_name=config['test_name'],
                    concurrent_loads=config['concurrent_loads'],
                    method=config['method'],
                    json_data=config.get('json_data'),
                    duration=15  # Shorter duration for faster testing
                )
                benchmark_results[config['test_name']] = results
                
            except Exception as e:
                logger.error(f"Error benchmarking {config['test_name']}: {e}")
                benchmark_results[config['test_name']] = {'error': str(e)}
        
        return benchmark_results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed performance report"""
        report = []
        report.append("# A-Share Trading System - API Performance Benchmark\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Target Server:** {self.base_url}")
        report.append("**Methodology:** Load testing with concurrent users over 15-second intervals\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        summary_stats = []
        for test_name, test_results in results.items():
            if 'error' in test_results:
                continue
                
            # Find best performance metrics
            for load_key, metrics in test_results.items():
                if isinstance(metrics, dict) and 'response_times' in metrics:
                    p95_ms = metrics['response_times']['p95'] * 1000
                    error_rate = metrics.get('error_rate', 0) * 100
                    throughput = metrics.get('throughput', 0)
                    
                    summary_stats.append({
                        'endpoint': test_name,
                        'load': load_key,
                        'p95_latency': p95_ms,
                        'error_rate': error_rate,
                        'throughput': throughput
                    })
        
        if summary_stats:
            # Find best performing configurations
            best_latency = min(summary_stats, key=lambda x: x['p95_latency'])
            best_throughput = max(summary_stats, key=lambda x: x['throughput'])
            
            report.append(f"- **Lowest Latency**: {best_latency['endpoint']} - {best_latency['p95_latency']:.1f}ms (P95)")
            report.append(f"- **Highest Throughput**: {best_throughput['endpoint']} - {best_throughput['throughput']:.1f} req/s")
            report.append(f"- **Overall System Status**: {'✅ Excellent' if best_latency['p95_latency'] < 100 else '⚠️ Needs Optimization'}")
        
        # Detailed Results
        report.append("\n## Detailed Performance Results\n")
        
        for test_name, test_results in results.items():
            if 'error' in test_results:
                report.append(f"### {test_name} - Error\n")
                report.append(f"**Error:** {test_results['error']}\n")
                continue
                
            report.append(f"### {test_name}\n")
            
            # Performance table
            report.append("| Concurrent Users | P50 (ms) | P95 (ms) | P99 (ms) | Error Rate | Throughput (req/s) |")
            report.append("|------------------|----------|----------|----------|------------|-------------------|")
            
            for load_key, metrics in test_results.items():
                if not isinstance(metrics, dict) or 'response_times' not in metrics:
                    continue
                    
                users = load_key.replace('_users', '')
                rt = metrics['response_times']
                
                report.append(f"| {users} | "
                            f"{rt['p50']*1000:.1f} | "
                            f"{rt['p95']*1000:.1f} | "
                            f"{rt['p99']*1000:.1f} | "
                            f"{metrics.get('error_rate', 0)*100:.1f}% | "
                            f"{metrics.get('throughput', 0):.1f} |")
            
            report.append("")
            
            # Additional metrics
            sample_metrics = next(iter(test_results.values()))
            if isinstance(sample_metrics, dict) and 'avg_content_length' in sample_metrics:
                report.append(f"**Average Response Size:** {sample_metrics['avg_content_length']:.0f} bytes\n")
        
        # Performance Analysis
        report.append("## Performance Analysis\n")
        
        # Latency Analysis
        report.append("### Latency Performance\n")
        report.append("| Endpoint | Status | P95 Latency | Recommendation |")
        report.append("|----------|--------|-------------|---------------|")
        
        for test_name, test_results in results.items():
            if 'error' in test_results:
                continue
                
            # Get single-user performance as baseline
            single_user = test_results.get('1_users', {})
            if 'response_times' in single_user:
                p95_ms = single_user['response_times']['p95'] * 1000
                
                if p95_ms < 100:
                    status = "✅ Excellent"
                    recommendation = "Performance meets SLA"
                elif p95_ms < 500:
                    status = "⚠️ Good"
                    recommendation = "Consider optimization"
                else:
                    status = "❌ Poor"
                    recommendation = "Requires immediate optimization"
                
                report.append(f"| {test_name} | {status} | {p95_ms:.1f}ms | {recommendation} |")
        
        # Scalability Analysis
        report.append("\n### Scalability Analysis\n")
        
        for test_name, test_results in results.items():
            if 'error' in test_results or len(test_results) < 2:
                continue
                
            report.append(f"**{test_name}:**\n")
            
            # Calculate scalability metrics
            load_points = []
            for load_key, metrics in test_results.items():
                if isinstance(metrics, dict) and 'response_times' in metrics:
                    users = int(load_key.replace('_users', ''))
                    p95_ms = metrics['response_times']['p95'] * 1000
                    error_rate = metrics.get('error_rate', 0) * 100
                    
                    load_points.append({
                        'users': users,
                        'p95': p95_ms,
                        'error_rate': error_rate
                    })
            
            if len(load_points) >= 2:
                load_points.sort(key=lambda x: x['users'])
                
                # Find breaking point
                breaking_point = None
                for point in load_points:
                    if point['p95'] > 1000 or point['error_rate'] > 5:  # 1s latency or 5% error rate
                        breaking_point = point['users']
                        break
                
                max_tested = load_points[-1]['users']
                
                if breaking_point:
                    report.append(f"- **Performance degrades** at {breaking_point} concurrent users")
                    report.append(f"- **Recommended max load**: {breaking_point * 0.8:.0f} concurrent users")
                else:
                    report.append(f"- **Stable performance** up to {max_tested} concurrent users")
                    report.append(f"- **Estimated capacity**: >{max_tested} concurrent users")
            
            report.append("")
        
        # Optimization Recommendations
        report.append("## Optimization Recommendations\n")
        
        recommendations = []
        
        # Analyze overall performance
        avg_p95_times = []
        max_error_rate = 0
        
        for test_name, test_results in results.items():
            if 'error' in test_results:
                continue
                
            for load_key, metrics in test_results.items():
                if isinstance(metrics, dict) and 'response_times' in metrics:
                    avg_p95_times.append(metrics['response_times']['p95'] * 1000)
                    max_error_rate = max(max_error_rate, metrics.get('error_rate', 0) * 100)
        
        avg_latency = statistics.mean(avg_p95_times) if avg_p95_times else 0
        
        if avg_latency > 500:
            recommendations.append("**Critical**: Average P95 latency exceeds 500ms - implement caching and database optimization")
        elif avg_latency > 100:
            recommendations.append("**High**: Consider Redis caching for frequently accessed data")
            
        if max_error_rate > 5:
            recommendations.append("**Critical**: Error rate exceeds 5% - investigate connection pooling and timeout settings")
        elif max_error_rate > 1:
            recommendations.append("**Medium**: Implement circuit breakers and retry mechanisms")
        
        # Generic recommendations
        recommendations.extend([
            "**Database**: Implement connection pooling and query optimization",
            "**Caching**: Deploy Redis cluster for stock data and computed indicators", 
            "**Load Balancing**: Use multiple FastAPI instances behind nginx/HAProxy",
            "**Async Processing**: Move heavy computations to background workers",
            "**CDN**: Cache static assets and API responses with appropriate TTL",
            "**Monitoring**: Implement APM with Prometheus/Grafana dashboards",
            "**Auto-scaling**: Deploy on Kubernetes with HPA based on CPU/memory",
            "**Database**: Migrate to PostgreSQL with read replicas for better concurrency"
        ])
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # SLA Compliance
        report.append("\n## SLA Compliance\n")
        report.append("**Target SLAs:**")
        report.append("- P95 Latency < 100ms for read operations")
        report.append("- P95 Latency < 500ms for compute operations") 
        report.append("- Error Rate < 0.1%")
        report.append("- Availability > 99.9%\n")
        
        report.append("**Compliance Status:**\n")
        
        compliant_endpoints = 0
        total_endpoints = 0
        
        for test_name, test_results in results.items():
            if 'error' in test_results:
                continue
                
            total_endpoints += 1
            single_user = test_results.get('1_users', {})
            
            if 'response_times' in single_user:
                p95_ms = single_user['response_times']['p95'] * 1000
                error_rate = single_user.get('error_rate', 0) * 100
                
                # Determine SLA target based on endpoint type
                sla_target = 500 if 'recommend' in test_name.lower() else 100
                
                if p95_ms <= sla_target and error_rate <= 0.1:
                    report.append(f"✅ **{test_name}**: Compliant (P95: {p95_ms:.1f}ms)")
                    compliant_endpoints += 1
                else:
                    report.append(f"❌ **{test_name}**: Non-compliant (P95: {p95_ms:.1f}ms, Errors: {error_rate:.1f}%)")
        
        if total_endpoints > 0:
            compliance_rate = compliant_endpoints / total_endpoints * 100
            report.append(f"\n**Overall SLA Compliance**: {compliance_rate:.1f}% ({compliant_endpoints}/{total_endpoints} endpoints)")
        
        report.append(f"\n---\n*Generated by API Performance Benchmark Suite*")
        
        return "\n".join(report)

async def main():
    """Main benchmark execution"""
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/")
            logger.info("✅ FastAPI server is running")
    except Exception as e:
        logger.error(f"❌ FastAPI server is not running: {e}")
        logger.error("Please start the server with: uvicorn api_server:app --host 0.0.0.0 --port 8000")
        return
    
    # Run benchmarks
    benchmark = APIBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # Generate and save report
    report = benchmark.generate_performance_report(results)
    
    os.makedirs("benchmarks", exist_ok=True)
    report_path = "benchmarks/api_performance.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save raw results
    json_path = "benchmarks/api_performance.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ API benchmark results saved to {report_path}")
    logger.info(f"✅ Raw data saved to {json_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("API PERFORMANCE BENCHMARK SUMMARY")
    print("="*70)
    
    for test_name, test_results in results.items():
        if 'error' in test_results:
            print(f"❌ {test_name}: {test_results['error']}")
            continue
            
        # Get single-user baseline
        baseline = test_results.get('1_users', {})
        if 'response_times' in baseline:
            p95_ms = baseline['response_times']['p95'] * 1000
            error_rate = baseline.get('error_rate', 0) * 100
            throughput = baseline.get('throughput', 0)
            
            print(f"\n✅ {test_name}:")
            print(f"   P95 Latency: {p95_ms:.1f}ms")
            print(f"   Error Rate: {error_rate:.1f}%") 
            print(f"   Throughput: {throughput:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(main())