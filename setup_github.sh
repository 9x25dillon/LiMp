#!/bin/bash

# Setup script for Entropy Engine GitHub repository
# This script helps initialize the repository and prepare it for GitHub

echo "🚀 Setting up Entropy Engine GitHub repository..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "feat: Initial release of Entropy Engine

- Core Token, EntropyNode, and EntropyEngine classes
- Command-line interface with built-in transformations
- Comprehensive test suite and documentation
- GitHub Actions CI/CD pipeline
- Package installation and distribution support"

echo "✅ Repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote origin: git remote add origin https://github.com/yourusername/entropy-engine.git"
echo "3. Push to GitHub: git push -u origin main"
echo "4. Update the repository URL in setup.py and pyproject.toml"
echo "5. Enable GitHub Actions in your repository settings"
echo ""
echo "🔧 To test the package locally:"
echo "   python3 -m entropy_engine.cli --input 'hello' --nodes 'root:reverse' --verbose"
echo ""
echo "🧪 To run tests:"
echo "   python3 run_tests.py"