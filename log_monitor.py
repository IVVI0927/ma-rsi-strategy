import time
from pathlib import Path
import json
from datetime import datetime, timedelta
import re
from collections import defaultdict
import logging
from logging_config import get_logger

logger = get_logger('monitor')

class LogMonitor:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.app_log = self.log_dir / "app.log"
        self.error_log = self.log_dir / "error.log"
        self.access_log = self.log_dir / "access.log"
        
        # Initialize file positions
        self.positions = {
            self.app_log: 0,
            self.error_log: 0,
            self.access_log: 0
        }
        
        # Initialize metrics
        self.metrics = {
            'error_count': 0,
            'request_count': 0,
            'avg_response_time': 0,
            'endpoint_usage': defaultdict(int),
            'status_codes': defaultdict(int)
        }
        
        # Compile regex patterns
        self.access_pattern = re.compile(r'Request processed.*?(\{.*\})', re.DOTALL)
        
    def read_new_lines(self, file_path: Path) -> list:
        """Read new lines from a file since last position."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(self.positions[file_path])
                new_lines = f.readlines()
                self.positions[file_path] = f.tell()
                return new_lines
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
    
    def parse_access_log(self, line: str) -> dict:
        """Parse a line from the access log."""
        try:
            match = self.access_pattern.search(line)
            if match:
                return json.loads(match.group(1))
            return None
        except Exception as e:
            logger.error(f"Error parsing access log line: {e}")
            return None
    
    def update_metrics(self, access_data: dict):
        """Update metrics based on access log data."""
        if not access_data:
            return
            
        self.metrics['request_count'] += 1
        
        # Update endpoint usage
        self.metrics['endpoint_usage'][access_data.get('path', 'unknown')] += 1
        
        # Update status codes
        self.metrics['status_codes'][str(access_data.get('status_code', 'unknown'))] += 1
        
        # Update average response time
        process_time = float(access_data.get('process_time', '0').rstrip('s'))
        current_avg = self.metrics['avg_response_time']
        self.metrics['avg_response_time'] = (current_avg * (self.metrics['request_count'] - 1) + process_time) / self.metrics['request_count']
    
    def check_errors(self, lines: list):
        """Check for new errors in the log."""
        for line in lines:
            if 'ERROR' in line:
                self.metrics['error_count'] += 1
                logger.warning(f"New error detected: {line.strip()}")
    
    def monitor(self, interval: int = 60):
        """Monitor logs at specified interval."""
        logger.info("Starting log monitoring...")
        
        while True:
            try:
                # Read new lines from all logs
                for log_file in [self.app_log, self.error_log, self.access_log]:
                    new_lines = self.read_new_lines(log_file)
                    
                    if log_file == self.access_log:
                        for line in new_lines:
                            access_data = self.parse_access_log(line)
                            if access_data:
                                self.update_metrics(access_data)
                    else:
                        self.check_errors(new_lines)
                
                # Print current metrics
                self.print_metrics()
                
                # Sleep for the specified interval
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def print_metrics(self):
        """Print current metrics."""
        print("\n=== Log Monitoring Metrics ===")
        print(f"Total Requests: {self.metrics['request_count']}")
        print(f"Total Errors: {self.metrics['error_count']}")
        print(f"Average Response Time: {self.metrics['avg_response_time']:.3f}s")
        
        print("\nTop Endpoints:")
        for endpoint, count in sorted(self.metrics['endpoint_usage'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {endpoint}: {count} requests")
        
        print("\nStatus Code Distribution:")
        for code, count in sorted(self.metrics['status_codes'].items()):
            print(f"  {code}: {count} requests")
        print("===========================\n")

if __name__ == "__main__":
    monitor = LogMonitor()
    monitor.monitor(interval=30)  # Check logs every 30 seconds 