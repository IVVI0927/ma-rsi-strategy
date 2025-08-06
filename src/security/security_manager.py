"""Security manager for air-gapped trading operations"""

import os
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import subprocess
import platform

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    NORMAL = "normal"
    TRADING = "trading"
    EMERGENCY = "emergency"

@dataclass
class SecurityEvent:
    event_type: str
    severity: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]

class EncryptionManager:
    """Handle data encryption and decryption"""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or self._generate_password()
        self.salt = os.urandom(16)
        self._key = self._derive_key(self.password, self.salt)
        self.fernet = Fernet(self._key)
        
    def _generate_password(self) -> str:
        """Generate secure random password"""
        return secrets.token_urlsafe(32)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data"""
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """Encrypt file"""
        output_path = output_path or f"{file_path}.encrypted"
        
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.encrypt_data(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(self.salt + encrypted_data)
        
        logger.info(f"File encrypted: {file_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """Decrypt file"""
        with open(encrypted_file_path, 'rb') as file:
            salt = file.read(16)
            encrypted_data = file.read()
        
        # Re-derive key with stored salt
        key = self._derive_key(self.password, salt)
        fernet = Fernet(key)
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        output_path = output_path or encrypted_file_path.replace('.encrypted', '')
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)
        
        logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
        return output_path

class NetworkManager:
    """Manage network connectivity for air-gapped operations"""
    
    def __init__(self):
        self.trading_mode = False
        self.allowed_hosts = []
        self.blocked_connections = []
        
    def enable_trading_mode(self):
        """Enable air-gapped trading mode"""
        logger.warning("Enabling trading mode - blocking external network access")
        self.trading_mode = True
        self._block_network_interfaces()
        
    def disable_trading_mode(self):
        """Disable trading mode and restore network access"""
        logger.info("Disabling trading mode - restoring network access")
        self.trading_mode = False
        self._restore_network_interfaces()
    
    def _block_network_interfaces(self):
        """Block external network interfaces during trading"""
        try:
            if platform.system() == "Darwin":  # macOS
                # Block external interfaces but keep localhost
                self._run_command(["sudo", "pfctl", "-e"])  # Enable packet filter
                # Add rules to block external traffic
                rules = """
                block out quick on en0
                block out quick on en1  
                pass out quick on lo0
                pass out quick to 127.0.0.0/8
                """
                with open("/tmp/pf_trading.conf", "w") as f:
                    f.write(rules)
                self._run_command(["sudo", "pfctl", "-f", "/tmp/pf_trading.conf"])
                
            elif platform.system() == "Linux":
                # Use iptables to block external connections
                self._run_command(["sudo", "iptables", "-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])
                self._run_command(["sudo", "iptables", "-A", "OUTPUT", "-j", "DROP"])
                
            logger.info("Network interfaces blocked for trading mode")
            
        except Exception as e:
            logger.error(f"Failed to block network interfaces: {e}")
    
    def _restore_network_interfaces(self):
        """Restore network interfaces after trading"""
        try:
            if platform.system() == "Darwin":  # macOS
                self._run_command(["sudo", "pfctl", "-d"])  # Disable packet filter
                
            elif platform.system() == "Linux":
                self._run_command(["sudo", "iptables", "-F", "OUTPUT"])
                
            logger.info("Network interfaces restored")
            
        except Exception as e:
            logger.error(f"Failed to restore network interfaces: {e}")
    
    def _run_command(self, command: List[str]) -> str:
        """Run system command safely"""
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {' '.join(command)}")
            return ""
        except Exception as e:
            logger.error(f"Command failed: {' '.join(command)}, error: {e}")
            return ""
    
    def check_network_status(self) -> Dict[str, Any]:
        """Check current network connectivity status"""
        interfaces = psutil.net_if_addrs()
        connections = psutil.net_connections()
        
        active_connections = [
            {
                'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                'status': conn.status,
                'pid': conn.pid
            }
            for conn in connections if conn.status == 'ESTABLISHED'
        ]
        
        return {
            'trading_mode': self.trading_mode,
            'interfaces': list(interfaces.keys()),
            'active_connections': active_connections,
            'external_connections': len([c for c in active_connections 
                                       if c['remote_address'] and not c['remote_address'].startswith('127.')])
        }

class AccessControlManager:
    """Manage access control and authentication"""
    
    def __init__(self, db_path: str = "security/access_control.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self.active_sessions = {}
        
    def _init_database(self):
        """Initialize access control database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    session_token TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    ip_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    success BOOLEAN
                )
            """)
    
    def create_user(self, username: str, password: str, role: str = "trader") -> bool:
        """Create new user"""
        try:
            salt = os.urandom(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    (username, base64.b64encode(password_hash).decode(), 
                     base64.b64encode(salt).decode(), role)
                )
                conn.commit()
                
            logger.info(f"User created: {username}")
            return True
            
        except sqlite3.IntegrityError:
            logger.error(f"User already exists: {username}")
            return False
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[str]:
        """Authenticate user and return session token"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, password_hash, salt, failed_attempts, locked_until FROM users WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                self._log_auth_attempt(None, "login", f"Unknown user: {username}", ip_address, False)
                return None
            
            user_id, stored_hash, salt, failed_attempts, locked_until = result
            
            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                self._log_auth_attempt(user_id, "login", "Account locked", ip_address, False)
                return None
            
            # Verify password
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', password.encode(), 
                base64.b64decode(salt.encode()), 100000
            )
            
            if base64.b64encode(password_hash).decode() == stored_hash:
                # Success - create session
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(hours=8)  # 8-hour sessions
                
                cursor.execute(
                    "INSERT INTO sessions (session_token, user_id, expires_at, ip_address) VALUES (?, ?, ?, ?)",
                    (session_token, user_id, expires_at, ip_address)
                )
                
                # Reset failed attempts
                cursor.execute(
                    "UPDATE users SET failed_attempts = 0, last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user_id,)
                )
                
                conn.commit()
                
                self.active_sessions[session_token] = {
                    'user_id': user_id,
                    'username': username,
                    'expires_at': expires_at,
                    'ip_address': ip_address
                }
                
                self._log_auth_attempt(user_id, "login", "Success", ip_address, True)
                logger.info(f"User authenticated: {username}")
                return session_token
                
            else:
                # Failed authentication
                new_failed_attempts = failed_attempts + 1
                locked_until = None
                
                if new_failed_attempts >= 5:  # Lock after 5 failed attempts
                    locked_until = datetime.now() + timedelta(minutes=30)
                
                cursor.execute(
                    "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                    (new_failed_attempts, locked_until, user_id)
                )
                conn.commit()
                
                self._log_auth_attempt(user_id, "login", f"Failed password (attempt {new_failed_attempts})", ip_address, False)
                return None
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token"""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            if datetime.now() < session['expires_at']:
                return session
            else:
                # Session expired
                del self.active_sessions[session_token]
        
        # Check database for session
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.user_id, u.username, s.expires_at, s.ip_address 
                FROM sessions s 
                JOIN users u ON s.user_id = u.id 
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
            """, (session_token,))
            result = cursor.fetchone()
            
            if result:
                user_id, username, expires_at, ip_address = result
                session = {
                    'user_id': user_id,
                    'username': username,
                    'expires_at': datetime.fromisoformat(expires_at),
                    'ip_address': ip_address
                }
                self.active_sessions[session_token] = session
                return session
        
        return None
    
    def logout_user(self, session_token: str):
        """Logout user and invalidate session"""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            self._log_auth_attempt(session['user_id'], "logout", "Success", session['ip_address'], True)
            del self.active_sessions[session_token]
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
            conn.commit()
    
    def _log_auth_attempt(self, user_id: Optional[int], action: str, details: str, 
                         ip_address: str, success: bool):
        """Log authentication attempt"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO audit_log (user_id, action, details, ip_address, success) VALUES (?, ?, ?, ?, ?)",
                (user_id, action, details, ip_address, success)
            )
            conn.commit()

class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.network_manager = NetworkManager()
        self.access_control = AccessControlManager()
        
        self.security_level = SecurityLevel.NORMAL
        self.security_events: List[SecurityEvent] = []
        
        # Security monitoring
        self.monitoring_enabled = True
        self.last_system_check = datetime.now()
        
    def enable_trading_mode(self) -> bool:
        """Enable secure trading mode"""
        try:
            logger.info("Enabling secure trading mode")
            
            # Switch to trading security level
            self.security_level = SecurityLevel.TRADING
            
            # Enable network restrictions
            self.network_manager.enable_trading_mode()
            
            # Encrypt sensitive data
            self._encrypt_sensitive_data()
            
            # Clear browser cache/history (basic implementation)
            self._clear_system_traces()
            
            # Log security event
            self._log_security_event(
                "trading_mode_enabled",
                "INFO",
                "Trading mode enabled - air-gapped operation active",
                {"timestamp": datetime.now(), "user": "system"}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable trading mode: {e}")
            self._log_security_event(
                "trading_mode_error",
                "ERROR",
                f"Failed to enable trading mode: {e}",
                {"error": str(e)}
            )
            return False
    
    def disable_trading_mode(self) -> bool:
        """Disable trading mode and restore normal operations"""
        try:
            logger.info("Disabling trading mode")
            
            # Restore network access
            self.network_manager.disable_trading_mode()
            
            # Switch back to normal security level
            self.security_level = SecurityLevel.NORMAL
            
            self._log_security_event(
                "trading_mode_disabled",
                "INFO",
                "Trading mode disabled - normal operations restored",
                {"timestamp": datetime.now()}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable trading mode: {e}")
            return False
    
    def _encrypt_sensitive_data(self):
        """Encrypt sensitive trading data"""
        sensitive_files = [
            "data/portfolio.csv",
            "data/positions.csv", 
            "logs/trades.log",
            "config/api_keys.json"
        ]
        
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                try:
                    self.encryption_manager.encrypt_file(file_path)
                    logger.debug(f"Encrypted: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to encrypt {file_path}: {e}")
    
    def _clear_system_traces(self):
        """Clear system traces and temporary data"""
        try:
            # Clear Python cache
            if os.path.exists("__pycache__"):
                import shutil
                shutil.rmtree("__pycache__", ignore_errors=True)
            
            # Clear temporary files
            temp_patterns = ["*.tmp", "*.cache", "*.log.*"]
            for pattern in temp_patterns:
                import glob
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                    except:
                        pass
            
            logger.debug("System traces cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear system traces: {e}")
    
    def perform_security_check(self) -> Dict[str, Any]:
        """Perform comprehensive security check"""
        security_status = {
            'timestamp': datetime.now(),
            'security_level': self.security_level.value,
            'trading_mode': self.network_manager.trading_mode,
            'checks': {}
        }
        
        # Network security check
        network_status = self.network_manager.check_network_status()
        security_status['checks']['network'] = {
            'status': 'secure' if network_status['trading_mode'] else 'normal',
            'external_connections': network_status['external_connections'],
            'details': network_status
        }
        
        # File system check
        sensitive_files_encrypted = self._check_file_encryption()
        security_status['checks']['encryption'] = {
            'status': 'encrypted' if sensitive_files_encrypted else 'plain',
            'encrypted_files': sensitive_files_encrypted
        }
        
        # Process security check
        suspicious_processes = self._check_suspicious_processes()
        security_status['checks']['processes'] = {
            'status': 'clean' if not suspicious_processes else 'suspicious',
            'suspicious_processes': suspicious_processes
        }
        
        # Memory security check
        memory_status = self._check_memory_security()
        security_status['checks']['memory'] = memory_status
        
        self.last_system_check = datetime.now()
        return security_status
    
    def _check_file_encryption(self) -> int:
        """Check how many sensitive files are encrypted"""
        encrypted_count = 0
        sensitive_files = ["data/portfolio.csv", "data/positions.csv", "logs/trades.log"]
        
        for file_path in sensitive_files:
            if os.path.exists(f"{file_path}.encrypted"):
                encrypted_count += 1
        
        return encrypted_count
    
    def _check_suspicious_processes(self) -> List[Dict[str, Any]]:
        """Check for suspicious processes"""
        suspicious = []
        
        # List of potentially suspicious process names
        suspicious_names = [
            'wireshark', 'tcpdump', 'netcat', 'nmap', 'burpsuite',
            'metasploit', 'sqlmap', 'john', 'hashcat'
        ]
        
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                if any(susp in proc.info['name'].lower() for susp in suspicious_names):
                    suspicious.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'user': proc.info['username']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return suspicious
    
    def _check_memory_security(self) -> Dict[str, Any]:
        """Check memory usage and security"""
        memory = psutil.virtual_memory()
        
        return {
            'status': 'normal' if memory.percent < 90 else 'high',
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }
    
    def _log_security_event(self, event_type: str, severity: str, message: str, details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log to file
        logger.log(
            logging.WARNING if severity == "WARNING" else 
            logging.ERROR if severity == "ERROR" else logging.INFO,
            f"Security Event [{severity}] {event_type}: {message}"
        )
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary"""
        recent_events = [e for e in self.security_events 
                        if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            'security_level': self.security_level.value,
            'trading_mode': self.network_manager.trading_mode,
            'recent_events': len(recent_events),
            'critical_events': len([e for e in recent_events if e.severity == "ERROR"]),
            'last_security_check': self.last_system_check,
            'active_sessions': len(self.access_control.active_sessions),
            'encryption_status': 'enabled' if self.security_level == SecurityLevel.TRADING else 'disabled'
        }

# Global security manager instance
security_manager = SecurityManager()