"""
Deployment Utilities for MarkItDown Testing Platform

Strategic deployment tools for various environments:
- Hugging Face Spaces optimization
- Local development setup
- Production environment configuration
- Health monitoring and diagnostics
"""

import os
import sys
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """Detect and configure for different deployment environments"""
    
    @staticmethod
    def detect_environment() -> str:
        """Detect the current deployment environment"""
        
        # Check for Hugging Face Spaces
        if os.environ.get('SPACE_ID'):
            return 'hf_spaces'
        
        # Check for Docker environment
        if os.path.exists('/.dockerenv'):
            return 'docker'
        
        # Check for common cloud providers
        if os.environ.get('HEROKU_APP_NAME'):
            return 'heroku'
        
        if os.environ.get('AWS_EXECUTION_ENV'):
            return 'aws'
        
        if os.environ.get('GOOGLE_CLOUD_PROJECT'):
            return 'gcp'
        
        # Default to local development
        return 'local'
    
    @staticmethod
    def get_environment_config(env_type: str) -> Dict[str, Any]:
        """Get configuration for specific environment"""
        
        configs = {
            'hf_spaces': {
                'max_file_size_mb': 50,
                'processing_timeout': 300,
                'max_memory_gb': 16,
                'temp_dir': '/tmp',
                'enable_analytics': True,
                'log_level': 'INFO',
                'gradio_config': {
                    'server_name': '0.0.0.0',
                    'server_port': 7860,
                    'share': False,
                    'enable_queue': True,
                    'max_file_size': '50mb'
                }
            },
            'docker': {
                'max_file_size_mb': 100,
                'processing_timeout': 600,
                'max_memory_gb': 32,
                'temp_dir': '/tmp',
                'enable_analytics': True,
                'log_level': 'INFO',
                'gradio_config': {
                    'server_name': '0.0.0.0',
                    'server_port': int(os.environ.get('PORT', 7860)),
                    'share': False,
                    'enable_queue': True,
                    'max_file_size': '100mb'
                }
            },
            'local': {
                'max_file_size_mb': 200,
                'processing_timeout': 900,
                'max_memory_gb': 64,
                'temp_dir': './temp',
                'enable_analytics': True,
                'log_level': 'DEBUG',
                'gradio_config': {
                    'server_name': '127.0.0.1',
                    'server_port': 7860,
                    'share': True,
                    'enable_queue': False,
                    'max_file_size': '200mb'
                }
            }
        }
        
        return configs.get(env_type, configs['local'])


class SystemHealthChecker:
    """System health monitoring and diagnostics"""
    
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Check system resource availability"""
        
        try:
            # Memory information
            memory = psutil.virtual_memory()
            
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # System information
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'architecture': platform.architecture()[0]
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'free_gb': memory.free / (1024**3)
                },
                'cpu': {
                    'count': cpu_count,
                    'usage_percent': cpu_percent,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'system': system_info,
                'status': 'healthy' if memory.percent < 80 and cpu_percent < 80 else 'warning'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """Check if all required dependencies are available"""
        
        required_packages = [
            'gradio',
            'markitdown',
            'google-genai',
            'plotly',
            'pandas',
            'numpy',
            'aiofiles',
            'tenacity',
            'psutil',
            'magic'
        ]
        
        dependency_status = {}
        all_available = True
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                dependency_status[package] = {'available': True, 'error': None}
            except ImportError as e:
                dependency_status[package] = {'available': False, 'error': str(e)}
                all_available = False
        
        return {
            'timestamp': datetime.now().isoformat(),
            'all_dependencies_available': all_available,
            'packages': dependency_status,
            'status': 'ready' if all_available else 'missing_dependencies'
        }
    
    @staticmethod
    def run_comprehensive_health_check() -> Dict[str, Any]:
        """Run comprehensive system health check"""
        
        logger.info("Starting comprehensive health check...")
        
        # Detect environment
        env_type = EnvironmentDetector.detect_environment()
        env_config = EnvironmentDetector.get_environment_config(env_type)
        
        # Check system resources
        resource_check = SystemHealthChecker.check_system_resources()
        
        # Check dependencies
        dependency_check = SystemHealthChecker.check_dependencies()
        
        # Overall health assessment
        overall_status = 'healthy'
        issues = []
        
        if resource_check.get('status') != 'healthy':
            overall_status = 'warning'
            issues.append('System resources under pressure')
        
        if not dependency_check.get('all_dependencies_available'):
            overall_status = 'error'
            issues.append('Missing required dependencies')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'type': env_type,
                'config': env_config
            },
            'system_resources': resource_check,
            'dependencies': dependency_check,
            'overall_status': overall_status,
            'issues': issues,
            'recommendations': SystemHealthChecker._generate_recommendations(
                env_type, resource_check, dependency_check
            )
        }
    
    @staticmethod
    def _generate_recommendations(
        env_type: str,
        resource_check: Dict[str, Any],
        dependency_check: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on health check results"""
        
        recommendations = []
        
        # Memory recommendations
        memory_percent = resource_check.get('memory', {}).get('used_percent', 0)
        if memory_percent > 80:
            recommendations.append("High memory usage detected. Consider reducing file sizes or processing batch sizes.")
        
        # CPU recommendations
        cpu_percent = resource_check.get('cpu', {}).get('usage_percent', 0)
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider enabling async processing or reducing concurrent operations.")
        
        # Environment-specific recommendations
        if env_type == 'hf_spaces':
            recommendations.extend([
                "Optimize for HF Spaces: Keep file sizes under 50MB",
                "Use stateless processing to avoid memory leaks",
                "Implement proper cleanup in temporary directories"
            ])
        
        # Dependency recommendations
        if not dependency_check.get('all_dependencies_available'):
            recommendations.append("Install missing dependencies using: pip install -r requirements.txt")
        
        return recommendations


class DeploymentConfigGenerator:
    """Generate configuration files for different deployment environments"""
    
    @staticmethod
    def generate_hf_spaces_config() -> Dict[str, str]:
        """Generate configuration files for HF Spaces"""
        
        # README.md content
        readme_content = """---
title: MarkItDown Testing Platform
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# MarkItDown Testing Platform

Enterprise-grade document conversion testing with AI-powered analysis.

## Features
- Multi-format document conversion
- Google Gemini AI analysis
- Interactive dashboards
- Quality metrics and reporting

## Usage
1. Upload your document
2. Configure analysis settings
3. Enter Gemini API key (optional)
4. Process and analyze results
"""
        
        # Dockerfile content
        dockerfile_content = """FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ libmagic1 libmagic-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
"""
        
        return {
            'README.md': readme_content,
            'Dockerfile': dockerfile_content
        }
    
    @staticmethod
    def save_deployment_configs(output_dir: str = "."):
        """Save all deployment configuration files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate HF Spaces configs
        hf_configs = DeploymentConfigGenerator.generate_hf_spaces_config()
        
        for filename, content in hf_configs.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated {filename} in {output_dir}")
        
        logger.info("All deployment configurations generated successfully")


class DeploymentValidator:
    """Validate deployment readiness"""
    
    @staticmethod
    def validate_for_hf_spaces() -> Dict[str, Any]:
        """Validate configuration for HF Spaces deployment"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'hf_spaces',
            'checks': {},
            'overall_status': 'ready',
            'issues': []
        }
        
        # Check required files
        required_files = ['app.py', 'requirements.txt', 'README.md']
        for file in required_files:
            if os.path.exists(file):
                validation_results['checks'][f'{file}_exists'] = True
            else:
                validation_results['checks'][f'{file}_exists'] = False
                validation_results['issues'].append(f"Missing required file: {file}")
                validation_results['overall_status'] = 'error'
        
        # Check app.py structure
        if os.path.exists('app.py'):
            try:
                with open('app.py', 'r') as f:
                    content = f.read()
                
                # Check for required components
                required_components = [
                    'gradio',
                    'launch',
                    'if __name__ == "__main__"'
                ]
                
                for component in required_components:
                    if component in content:
                        validation_results['checks'][f'app_{component}'] = True
                    else:
                        validation_results['checks'][f'app_{component}'] = False
                        validation_results['issues'].append(f"Missing component in app.py: {component}")
                        validation_results['overall_status'] = 'warning'
                        
            except Exception as e:
                validation_results['checks']['app_readable'] = False
                validation_results['issues'].append(f"Cannot read app.py: {e}")
                validation_results['overall_status'] = 'error'
        
        # Check requirements.txt
        if os.path.exists('requirements.txt'):
            try:
                with open('requirements.txt', 'r') as f:
                    requirements = f.read()
                
                # Check for essential packages
                essential_packages = ['gradio', 'markitdown', 'google-genai']
                for package in essential_packages:
                    if package in requirements:
                        validation_results['checks'][f'req_{package}'] = True
                    else:
                        validation_results['checks'][f'req_{package}'] = False
                        validation_results['issues'].append(f"Missing package in requirements.txt: {package}")
                        validation_results['overall_status'] = 'warning'
                        
            except Exception as e:
                validation_results['checks']['requirements_readable'] = False
                validation_results['issues'].append(f"Cannot read requirements.txt: {e}")
                validation_results['overall_status'] = 'error'
        
        # Check file sizes (HF Spaces has limits)
        total_size = 0
        for root, dirs, files in os.walk('.'):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        total_size_mb = total_size / (1024 * 1024)
        validation_results['checks']['total_size_mb'] = total_size_mb
        
        if total_size_mb > 500:  # HF Spaces limit
            validation_results['issues'].append(f"Total size ({total_size_mb:.2f}MB) exceeds HF Spaces limit")
            validation_results['overall_status'] = 'error'
        
        return validation_results
    
    @staticmethod
    def generate_deployment_report() -> str:
        """Generate comprehensive deployment readiness report"""
        
        # Run health check
        health_check = SystemHealthChecker.run_comprehensive_health_check()
        
        # Run HF Spaces validation
        hf_validation = DeploymentValidator.validate_for_hf_spaces()
        
        # Generate report
        report = f"""
# MarkItDown Testing Platform - Deployment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Environment Information
- **Type**: {health_check['environment']['type']}
- **Platform**: {health_check['system_resources']['system']['platform']}
- **Python**: {health_check['system_resources']['system']['python_version']}

## System Health Status: {health_check['overall_status'].upper()}

### System Resources
- **Memory**: {health_check['system_resources']['memory']['available_gb']:.2f}GB available ({health_check['system_resources']['memory']['used_percent']:.1f}% used)
- **CPU**: {health_check['system_resources']['cpu']['count']} cores, {health_check['system_resources']['cpu']['usage_percent']:.1f}% usage
- **Disk**: {health_check['system_resources']['disk']['free_gb']:.2f}GB free

### Dependencies Status: {"✅ READY" if health_check['dependencies']['all_dependencies_available'] else "❌ MISSING"}
"""
        
        # Add dependency details
        for package, status in health_check['dependencies']['packages'].items():
            status_icon = "✅" if status['available'] else "❌"
            report += f"- {status_icon} {package}\n"
        
        # Add HF Spaces validation
        report += f"""
## HF Spaces Deployment Readiness: {hf_validation['overall_status'].upper()}

### File Validation
"""
        
        for check, result in hf_validation['checks'].items():
            status_icon = "✅" if result else "❌"
            report += f"- {status_icon} {check}\n"
        
        # Add issues and recommendations
        if health_check['issues']:
            report += "\n### Issues Identified:\n"
            for issue in health_check['issues']:
                report += f"- ⚠️ {issue}\n"
        
        if hf_validation['issues']:
            report += "\n### HF Spaces Issues:\n"
            for issue in hf_validation['issues']:
                report += f"- ⚠️ {issue}\n"
        
        if health_check['recommendations']:
            report += "\n### Recommendations:\n"
            for rec in health_check['recommendations']:
                report += f"- 💡 {rec}\n"
        
        # Add deployment commands
        report += f"""
## Deployment Commands

### Local Development
```bash
python app.py
```

### Docker Deployment
```bash
docker build -t markitdown-platform .
docker run -p 7860:7860 markitdown-platform
```

### HF Spaces Deployment
1. Create new Space on Hugging Face
2. Upload files or connect GitHub repository
3. Configure Space settings:
   - SDK: Gradio
   - Python version: 3.10
   - Hardware: CPU (free tier)

---
Report generated by MarkItDown Testing Platform Deployment Utils
"""
        
        return report


def main():
    """Main function for deployment utilities CLI"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MarkItDown Platform Deployment Utilities')
    parser.add_argument(
        'command',
        choices=['health-check', 'validate', 'generate-configs', 'report'],
        help='Command to execute'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='text',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    if args.command == 'health-check':
        result = SystemHealthChecker.run_comprehensive_health_check()
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"System Status: {result['overall_status'].upper()}")
            print(f"Environment: {result['environment']['type']}")
            print(f"Memory Available: {result['system_resources']['memory']['available_gb']:.2f}GB")
            print(f"Dependencies: {'OK' if result['dependencies']['all_dependencies_available'] else 'MISSING'}")
            
            if result['issues']:
                print("\nIssues:")
                for issue in result['issues']:
                    print(f"  - {issue}")
    
    elif args.command == 'validate':
        result = DeploymentValidator.validate_for_hf_spaces()
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"HF Spaces Validation: {result['overall_status'].upper()}")
            if result['issues']:
                print("Issues found:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            else:
                print("✅ Ready for HF Spaces deployment!")
    
    elif args.command == 'generate-configs':
        DeploymentConfigGenerator.save_deployment_configs(args.output)
        print(f"Configuration files generated in {args.output}")
    
    elif args.command == 'report':
        report = DeploymentValidator.generate_deployment_report()
        
        if args.output != '.':
            os.makedirs(args.output, exist_ok=True)
            report_file = os.path.join(args.output, 'deployment_report.md')
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"Deployment report saved to {report_file}")
        else:
            print(report)


if __name__ == "__main__":
    main()
