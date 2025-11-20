# Contributing to Embeddings at Scale

Thank you for contributing! This guide ensures your changes pass CI/CD checks.

## Quick Start

```bash
# Install dependencies and setup hooks
make setup

# Run all CI/CD checks locally before committing
make ci-check
```

## Development Workflow

### 1. Setup Environment

```bash
# Install all dependencies with correct versions
make setup
```

This installs:
- `ruff==0.14.4` (matches CI/CD version)
- Pre-commit hooks
- All Python dependencies

### 2. Make Changes

Edit code in `code_examples/` directory.

### 3. Check Before Committing

Run CI/CD checks locally:

```bash
# Run all checks (recommended)
make ci-check

# Or run individual checks
make lint      # Ruff linter with auto-fix
make format    # Ruff formatter
make test      # Python syntax check
```

### 4. Commit

Pre-commit hooks will automatically run when you commit. If they fail, fix the issues and commit again.

```bash
git add .
git commit -m "Your message"
```

## CI/CD Checks

All pull requests must pass these checks:

1. **Ruff Linter** - Code quality and style
2. **Ruff Formatter** - Code formatting
3. **Python Syntax** - All files must compile
4. **Dependencies** - Requirements must be valid

## Tool Versions

**IMPORTANT**: Always use the same versions as CI/CD:

- `ruff==0.14.4`
- Python 3.11

The `make setup` command installs the correct versions automatically.

## For Claude Code Users

When Claude Code starts a session, it automatically:
- Installs `ruff==0.14.4`
- Ensures environment matches CI/CD

This is configured in `.claude/SessionStart`.

## Common Issues

### "Would reformat" error

```bash
# Fix formatting issues
make format
git add .
git commit --amend --no-edit
```

### Linter errors

```bash
# Auto-fix lint issues
make lint
```

### Version mismatch

```bash
# Reinstall with correct versions
make setup
```

## File Structure

```
.
├── .claude/
│   └── SessionStart          # Auto-setup for Claude Code
├── .pre-commit-config.yaml   # Pre-commit hooks config
├── scripts/
│   └── ci-check.sh          # Local CI/CD check script
├── Makefile                  # Development commands
└── code_examples/            # All code examples
```

## Questions?

Check the CI/CD configuration:
- `.github/workflows/code-quality.yml` - CI/CD pipeline
- `.pre-commit-config.yaml` - Pre-commit hooks
- `Makefile` - Development commands
