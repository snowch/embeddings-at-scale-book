#!/bin/bash
# Run the same checks as CI/CD locally
# This ensures your changes will pass CI before pushing

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Running CI/CD checks locally..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check ruff version
RUFF_VERSION=$(ruff --version | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
EXPECTED_VERSION="0.14.4"

if [ "$RUFF_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "⚠️  Warning: ruff version mismatch"
    echo "   Expected: $EXPECTED_VERSION"
    echo "   Found: $RUFF_VERSION"
    echo "   Installing correct version..."
    pip install -q ruff==$EXPECTED_VERSION
fi

# 1. Ruff linter
echo "📋 Running ruff linter..."
if ruff check code_examples/; then
    echo "✅ Ruff linter passed"
else
    echo "❌ Ruff linter failed"
    exit 1
fi
echo ""

# 2. Ruff formatter check
echo "🎨 Running ruff formatter check..."
if ruff format --check code_examples/; then
    echo "✅ Ruff formatter passed"
else
    echo "❌ Ruff formatter failed"
    echo "   Run: ruff format code_examples/ to fix"
    exit 1
fi
echo ""

# 3. Python syntax check
echo "🐍 Running Python syntax check..."
total=0
errors=0

for file in $(find code_examples -name "*.py" -type f); do
    total=$((total + 1))
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        echo "❌ SYNTAX ERROR: $file"
        python3 -m py_compile "$file" 2>&1 || true
        errors=$((errors + 1))
    fi
done

if [ $errors -gt 0 ]; then
    echo "❌ Syntax check failed: $errors file(s) have errors"
    exit 1
else
    echo "✅ Python syntax check passed ($total files)"
fi
echo ""

# 4. Dependencies check
echo "📦 Running dependencies check..."
if [ ! -f "code_examples/requirements.txt" ]; then
    echo "❌ code_examples/requirements.txt not found"
    exit 1
fi
echo "✅ Dependencies check passed"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All CI/CD checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
