# Code Linting Setup

This repository uses **ruff** for fast Python linting and code quality checks.

## Quick Start

### Install Pre-commit Hooks (Recommended)

For automatic linting on every commit:

```bash
pip install pre-commit
pre-commit install
```

Now linting will run automatically when you commit changes.

### Manual Linting

Check code quality:

```bash
# Check all code
ruff check code_examples/

# Check specific file
ruff check code_examples/ch05_contrastive_learning/infonceloss.py

# Auto-fix issues
ruff check code_examples/ --fix
```

Format code:

```bash
# Check formatting
ruff format --check code_examples/

# Auto-format code
ruff format code_examples/
```

## CI/CD Integration

The repository includes automated code quality checks via GitHub Actions:

- **Workflow**: `.github/workflows/code-quality.yml`
- **Triggers**: Push to main, push to claude/* branches, pull requests
- **Checks**:
  - Ruff linting
  - Python syntax validation
  - Dependencies verification

View workflow results: Actions tab on GitHub

## Current Code Quality Status

**Last Updated**: 2025-11-19

### Syntax Check
- ✅ **253/253 files** compile successfully
- ✅ **100% success rate**

### Linting Analysis

Total linting issues detected: **6,709**

**Auto-fixable**: 4,339 issues (65%)

#### Issue Breakdown

| Issue Type | Count | Fixable | Severity |
|------------|-------|---------|----------|
| Unused imports | 202 | ✅ Yes | Low |
| F-string without placeholders | 189 | ✅ Yes | Low |
| Unsorted imports | 131 | ✅ Yes | Low |
| Trailing whitespace | 72 | ✅ Yes | Low |
| Unused variables | 50 | ❌ No | Low |
| Undefined names | 1,444 | ❌ No | Expected* |
| Other style issues | ~4,621 | Mixed | Low |

\* **Note on undefined names**: Many code examples are educational snippets extracted from chapters. They intentionally focus on specific concepts and may not include all imports. This is normal for a technical book.

## Configuration

### Ruff Configuration

Location: `pyproject.toml`

Key settings:
- **Line length**: 100 characters
- **Target Python**: 3.8+
- **Enabled rules**: pycodestyle, pyflakes, isort, pep8-naming, flake8-bugbear
- **Ignored rules**: E501 (line too long), N803/N806 (ML naming conventions)

### Pre-commit Configuration

Location: `.pre-commit-config.yaml`

Hooks:
- Ruff linter and formatter
- Python AST validation
- YAML/TOML validation
- Trailing whitespace removal
- Bandit security checks

## Auto-fixing Issues

### Safe Auto-fixes

These can be safely applied to all code:

```bash
# Fix import sorting
ruff check code_examples/ --select I --fix

# Remove trailing whitespace
ruff check code_examples/ --select W291 --fix

# Fix f-string issues
ruff check code_examples/ --select F541 --fix

# Remove unused imports (review first!)
ruff check code_examples/ --select F401 --fix
```

### Bulk Auto-fix (Use with caution)

To auto-fix all fixable issues:

```bash
# Review changes first!
ruff check code_examples/ --fix --diff

# Apply fixes
ruff check code_examples/ --fix
```

**⚠️ Warning**: Review changes before committing. Some auto-fixes may remove code that's intentionally present for educational purposes.

## Customizing Rules

### Ignore Specific Issues

Add to `pyproject.toml`:

```toml
[tool.ruff.lint]
ignore = [
    "E501",  # Line too long
    "F821",  # Undefined name (for educational snippets)
]
```

### Per-file Ignores

```toml
[tool.ruff.lint.per-file-ignores]
"code_examples/ch01_*/*.py" = ["F821"]  # Allow undefined names in Ch01
```

### Disable Linting in Code

```python
# ruff: noqa: F821
def example():
    result = undefined_function()  # Won't trigger F821 error
```

## Best Practices

### For Contributors

1. **Install pre-commit hooks** before making changes
2. **Run linting locally** before pushing
3. **Review auto-fixes** to ensure they don't break educational content
4. **Use inline ignores** for intentional violations

### For Maintainers

1. **Monitor CI/CD status** on pull requests
2. **Gradually improve code quality** over time
3. **Update ruff version** periodically for new features
4. **Document exceptions** when disabling rules

## Troubleshooting

### Pre-commit Hook Fails

```bash
# Skip hooks temporarily
git commit --no-verify -m "message"

# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean
```

### Ruff Errors

```bash
# Show detailed error information
ruff check code_examples/ --verbose

# Generate report
ruff check code_examples/ --output-format=json > lint-report.json
```

### CI/CD Workflow Fails

1. Check Actions tab for detailed error logs
2. Run same checks locally: `ruff check code_examples/`
3. Ensure ruff version matches CI: `pip install ruff==0.1.9`

## Maintenance

### Update Ruff

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Update pip installation
pip install --upgrade ruff
```

### Update CI/CD

Edit `.github/workflows/code-quality.yml` to change:
- Ruff version
- Python version
- Enabled checks

## Resources

- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **Pre-commit Documentation**: https://pre-commit.com/
- **GitHub Actions**: https://docs.github.com/en/actions

## Status Badge

Add to README.md:

```markdown
![Code Quality](https://github.com/snowch/embeddings-at-scale-book/actions/workflows/code-quality.yml/badge.svg)
```

---

**Summary**: Linting infrastructure is fully set up and operational. Code is syntactically valid (100%). Style issues are documented and can be gradually improved. All automation is in place for continuous quality monitoring.
