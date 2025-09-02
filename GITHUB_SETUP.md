# GitHub Repository Setup Guide

This guide will help you set up the Entropy Engine project on GitHub.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   ./setup_github.sh
   ```

2. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name it `entropy-engine`
   - Make it public or private as preferred
   - Don't initialize with README (we already have one)

3. **Connect your local repository to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/entropy-engine.git
   git branch -M main
   git push -u origin main
   ```

4. **Update repository URLs:**
   Edit these files and replace `yourusername` with your actual GitHub username:
   - `setup.py` (line with `url=`)
   - `pyproject.toml` (all URLs in `[project.urls]`)

5. **Enable GitHub Actions:**
   - Go to your repository settings
   - Navigate to Actions > General
   - Enable "Allow all actions and reusable workflows"

## 📁 Repository Structure

```
entropy-engine/
├── entropy_engine/           # Main package
│   ├── __init__.py
│   ├── core.py              # Core classes
│   └── cli.py               # Command-line interface
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_core.py
├── .github/                 # GitHub configuration
│   ├── workflows/
│   │   └── ci.yml          # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── docs/                    # Documentation
├── examples/                # Example scripts
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # Version history
├── .gitignore            # Git ignore rules
├── run_tests.py          # Simple test runner
└── setup_github.sh       # Setup script
```

## 🧪 Testing

### Run the simple test suite:
```bash
python3 run_tests.py
```

### Test the CLI:
```bash
# List available transformations
python3 -m entropy_engine.cli --list-transforms

# Process a simple example
python3 -m entropy_engine.cli --input "hello" --nodes "root:reverse" --verbose

# Save results to file
python3 -m entropy_engine.cli --input "test" --nodes "root:uppercase" --output results.json
```

## 📦 Package Installation

### Development installation:
```bash
pip install -e .
```

### Install with development dependencies:
```bash
pip install -e ".[dev]"
```

### Build distribution:
```bash
python -m build
```

## 🔧 Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make your changes and test:**
   ```bash
   python3 run_tests.py
   python3 -m entropy_engine.cli --input "test" --nodes "root:reverse"
   ```

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **Push and create a pull request:**
   ```bash
   git push origin feature/new-feature
   ```

## 🎯 GitHub Features

### Issues
- Use the provided templates for bug reports and feature requests
- Include code examples and environment details

### Pull Requests
- Use the provided PR template
- Ensure all tests pass
- Update documentation if needed

### Actions
- CI pipeline runs on every push and PR
- Tests multiple Python versions
- Checks code formatting and linting
- Builds the package

## 📚 Documentation

- **README.md**: Main project documentation
- **CONTRIBUTING.md**: How to contribute
- **CHANGELOG.md**: Version history
- **Code comments**: Inline documentation

## 🚀 Publishing to PyPI

When ready to publish:

1. **Update version numbers:**
   - `entropy_engine/__init__.py`
   - `setup.py`
   - `pyproject.toml`
   - `CHANGELOG.md`

2. **Build and upload:**
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## 🆘 Getting Help

- Check the existing issues on GitHub
- Create a new issue with the provided template
- Review the documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.