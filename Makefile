# Makefile for Embeddings at Scale Book
# Makes it easy to run common development tasks

.PHONY: help setup ci-check lint format test clean

# Default target
help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📚 Embeddings at Scale - Development Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "setup        - Install development dependencies"
	@echo "ci-check     - Run all CI/CD checks locally"
	@echo "lint         - Run ruff linter (with auto-fix)"
	@echo "format       - Run ruff formatter"
	@echo "test         - Run Python syntax check"
	@echo "clean        - Remove Python cache files"
	@echo ""

# Install dependencies with CI/CD versions
setup:
	@echo "📦 Installing dependencies..."
	pip install ruff==0.14.4
	pip install pre-commit
	pip install -r code_examples/requirements.txt
	pre-commit install
	@echo "✅ Setup complete!"

# Run all CI/CD checks
ci-check:
	@./scripts/ci-check.sh

# Run linter
lint:
	@echo "📋 Running ruff linter..."
	ruff check code_examples/ --fix

# Run formatter
format:
	@echo "🎨 Running ruff formatter..."
	ruff format code_examples/

# Run syntax check
test:
	@echo "🐍 Running Python syntax check..."
	@for file in $$(find code_examples -name "*.py" -type f); do \
		python3 -m py_compile "$$file" || exit 1; \
	done
	@echo "✅ All files compile successfully!"

# Clean up cache files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Clean complete!"
