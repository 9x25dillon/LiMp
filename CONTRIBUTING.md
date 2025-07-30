# Contributing to Entropy Engine

Thank you for your interest in contributing to Entropy Engine! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
5. Install the package in development mode: `pip install -e .`
6. Install development dependencies: `pip install -e ".[dev]"`

## Development Setup

### Prerequisites
- Python 3.7 or higher
- pip
- git

### Code Style
We use the following tools to maintain code quality:
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

Run these tools before submitting:
```bash
black entropy_engine/
flake8 entropy_engine/
mypy entropy_engine/
```

### Testing
We use pytest for testing. Run tests with:
```bash
pytest tests/ -v
```

For coverage:
```bash
pytest tests/ --cov=entropy_engine --cov-report=html
```

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Commit your changes with a descriptive message

### Commit Message Format
Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the CHANGELOG.md with a note describing your changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages or logs

## Feature Requests

When requesting features, please include:
- A clear description of the feature
- Use cases and examples
- Any alternative solutions you've considered

## Code of Conduct

This project is committed to providing a welcoming and inspiring community for all. Please be respectful and inclusive in all interactions.

## License

By contributing to Entropy Engine, you agree that your contributions will be licensed under the MIT License.