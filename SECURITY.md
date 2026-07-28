# Security policy

## Supported versions

Security fixes are applied to the latest tagged release.

## Reporting a vulnerability

Do not disclose a serious vulnerability in a public issue. Contact the repository maintainers privately through the security-advisory interface after the repository is published.

## Checkpoint safety

PyTorch checkpoint loading can deserialize Python objects. Load only checkpoints obtained from trusted release pages and verify the published SHA-256 checksum.

## Sensitive data

Never commit credentials, access tokens, private dataset URLs, institutional account details, or restricted imagery. Git history remains accessible even after a file is removed from the latest revision unless the history is rewritten.
