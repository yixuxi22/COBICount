# Contributing

Thank you for improving COBICount.

## Before opening an issue

- search existing issues;
- include the repository version or commit hash;
- include Python, PyTorch, CUDA, GPU, and operating-system information;
- provide the full command and a minimal traceback;
- remove private paths, credentials, dataset content, and personal information.

## Development setup

```bash
python -m venv .venv
# activate the environment
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Pull requests

1. Create a focused branch.
2. Keep changes small and explain the research/protocol impact.
3. Add or update tests when behavior changes.
4. Do not add target-domain access to the training entry point.
5. Do not commit datasets, model weights, generated runs, or local absolute paths.
6. Update documentation and `CHANGELOG.md` when relevant.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
