# Trunk Configuration

This directory contains configuration files and tools for [Trunk](https://trunk.io), our unified code quality and linting tool.

## Directory Structure

```
.trunk/
├── configs/              # Language-specific linter configurations
│   ├── .markdownlint.yaml   # Markdown linting rules
│   └── .yamllint.yaml       # YAML linting rules
├── out/                 # Generated output files (gitignored)
├── logs/                # Trunk operation logs (gitignored)
├── notifications/       # Trunk notification data (gitignored)
├── plugins/            # Trunk plugins (gitignored)
├── tools/              # Cached tool installations (gitignored)
└── trunk.yaml          # Main Trunk configuration
```

## Configuration

Our Trunk setup includes:

### Linters

- **Python**

  - `black` - Code formatting
  - `isort` - Import sorting
  - `mypy` - Static type checking
  - `ruff` - Fast Python linter
  - `bandit` - Security linting

- **Documentation & Config**

  - `markdownlint` - Markdown formatting
  - `yamllint` - YAML validation
  - `prettier` - General formatting

- **Security**
  - `trufflehog` - Secret scanning
  - `git-diff-check` - Whitespace error detection

### Actions

- `trunk-announce` - Notify about Trunk events
- `trunk-check-pre-push` - Run checks before git push
- `trunk-fmt-pre-commit` - Format code before commits
- `trunk-upgrade-available` - Check for Trunk updates

## Usage

Common commands:

```bash
# Check all files
trunk check --all

# Format all files
trunk fmt --all

# Check specific files
trunk check path/to/file.py

# Run specific linter
trunk check --all --filter=black
```

## Integration

Trunk is integrated with our CI/CD pipeline through GitHub Actions:

- Pre-commit hooks for code formatting
- Pre-push hooks for comprehensive checks
- Automated PR checks with inline annotations
- Caching for faster CI runs

## Cache Management

Trunk caches tools and configurations in:

- `.trunk/tools/` - Linter binaries
- `.trunk/out/` - Linter outputs
- `~/.cache/trunk` - Global cache (CI uses this)

All cache directories are gitignored.

## Updating

To update Trunk and its tools:

```bash
trunk upgrade --all
```

Configuration changes should be made in:

- `trunk.yaml` for tool versions and general settings
- `.trunk/configs/` for language-specific rules
