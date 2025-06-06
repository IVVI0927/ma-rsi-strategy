# Development Environment Setup Guide

## Prerequisites

- Python 3.11 or higher
- Git
- Poetry (Python package manager)
- Docker (optional, for containerized development)
- VS Code (recommended IDE)

## Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ma_agent_project.git
cd ma_agent_project
```

2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Configure Poetry:
```bash
poetry config virtualenvs.in-project true
poetry config virtualenvs.path .venv
```

4. Install dependencies:
```bash
poetry install
```

5. Create and configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` with your API keys and configuration.

## Development Workflow

### 1. Code Style and Quality

- Use Black for code formatting:
```bash
poetry run black .
```

- Use Flake8 for linting:
```bash
poetry run flake8
```

- Run type checking:
```bash
poetry run mypy .
```

### 2. Testing

- Run tests:
```bash
poetry run pytest
```

- Run tests with coverage:
```bash
poetry run pytest --cov=src tests/
```

### 3. Documentation

- Generate API documentation:
```bash
poetry run pdoc src/ --output-dir docs/api
```

- Update README:
```bash
poetry run python scripts/update_readme.py
```

## IDE Setup

### VS Code

1. Install recommended extensions:
   - Python
   - Pylance
   - Python Test Explorer
   - Python Docstring Generator
   - GitLens

2. Configure VS Code settings:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Configure project interpreter:
   - File -> Settings -> Project -> Python Interpreter
   - Add new interpreter -> Poetry Environment
   - Select the project's `.venv` directory

2. Enable code style:
   - File -> Settings -> Editor -> Code Style
   - Set Python code style to follow PEP 8

## Docker Development

1. Build the development container:
```bash
docker build -t ma_agent_dev -f Dockerfile.dev .
```

2. Run the container:
```bash
docker run -it --rm -v $(pwd):/app ma_agent_dev
```

## Common Development Tasks

### Adding New Dependencies

```bash
poetry add package_name
poetry add --group dev package_name  # for development dependencies
```

### Updating Dependencies

```bash
poetry update
```

### Running the Application

```bash
poetry run python src/main.py
```

### Running Tests

```bash
poetry run pytest
```

### Generating Documentation

```bash
poetry run pdoc src/ --output-dir docs/api
```

## Troubleshooting

### Common Issues

1. Poetry installation fails:
   - Ensure Python 3.11+ is installed
   - Try installing Poetry with pip: `pip install poetry`

2. Virtual environment issues:
   - Delete `.venv` directory
   - Run `poetry install` again

3. Test failures:
   - Check test data files exist
   - Verify environment variables are set correctly

### Getting Help

- Check the [GitHub Issues](https://github.com/yourusername/ma_agent_project/issues)
- Join the [Discord community](https://discord.gg/your-server)
- Contact the maintainers

## Contributing

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes

3. Run tests and linting:
```bash
poetry run pytest
poetry run flake8
```

4. Commit your changes:
```bash
git commit -m "feat: your feature description"
```

5. Push to GitHub:
```bash
git push origin feature/your-feature-name
```

6. Create a Pull Request

## Deployment

### Local Deployment

1. Build the application:
```bash
poetry build
```

2. Run the application:
```bash
poetry run python src/main.py
```

### Production Deployment

1. Build the Docker image:
```bash
docker build -t ma_agent:latest .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 ma_agent:latest
```

## Monitoring and Logging

- Logs are stored in `logs/` directory
- Use `poetry run python scripts/monitor.py` to view real-time logs
- Set up alerts in `config/alerts.yaml` 