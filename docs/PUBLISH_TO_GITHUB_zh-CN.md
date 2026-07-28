# COBICount 发布到 GitHub 的完整流程

## 0. 发布前先确认

本仓库已经包含 MIT `LICENSE`、`README.md`、`CITATION.cff`、贡献规范、Issue 模板和 GitHub Actions。正式公开前仍需人工完成以下事项：

1. 全局替换 `OWNER/COBICount` 为你的真实 GitHub 用户名或组织名，例如 `YOUR_USERNAME/COBICount`。重点文件：
   - `README.md`
   - `pyproject.toml`
   - `CITATION.cff`
   - `.github/ISSUE_TEMPLATE/config.yml`
2. 确认作者姓名、软件版本和 MIT 许可证符合所有共同作者的决定。
3. 确认投稿期刊允许在当前审稿阶段公开带作者身份的代码。若仍需匿名，应先删除作者身份并使用匿名仓库。
4. 不要把数据集、`.pth` 权重、本地运行结果或绝对路径提交到 Git 历史。

## 1. 本地检查

在项目根目录运行：

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
python tools/release_check.py
pytest -q
```

`release_check.py` 会检查目标域数据类是否重新进入训练入口、是否存在权重/数据文件、明显的原始本地路径和未替换的 `OWNER` 占位符。

## 2. 在 GitHub 创建空仓库

建议仓库名称使用 `COBICount`。创建时选择 Public。由于本地包中已经包含 README、LICENSE 和 `.gitignore`，网页创建仓库时不要再次初始化这些文件，避免首次推送产生无关冲突。

## 3. 使用 Git 命令首次推送

在解压后的 `COBICount` 根目录执行：

```bash
git init
git add .
git status
git commit -m "Initial public release of COBICount"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/COBICount.git
git push -u origin main
```

也可以使用 GitHub CLI：

```bash
gh auth login
gh repo create COBICount --public --source=. --remote=origin --push
```

提交前必须查看 `git status`，确认没有数据集、权重、运行结果或隐私文件被暂存。

## 4. 设置仓库主页

进入仓库首页，在 About 区域填写：

- Description：`Official PyTorch implementation of COBICount for source-only remote sensing object counting.`
- Topics：`remote-sensing`、`object-counting`、`domain-generalization`、`source-only`、`pytorch`
- Website：论文页面、预印本或项目主页；尚未公开时可留空。

然后检查：

- README 是否正常渲染；
- GitHub 是否识别 MIT License；
- 右侧是否出现 “Cite this repository”；
- Actions 中 CI 是否通过；
- Issues 模板是否正常出现。

## 5. 建议的安全设置

在 `Settings` 中检查：

- `Actions`：允许仓库中的工作流运行；
- `Code security and analysis`：启用 Dependabot alerts；可用时启用 secret scanning 和 push protection；
- `Branches`：多人协作时为 `main` 添加分支保护，要求 Pull Request 和 CI 通过后再合并；
- `General`：确认仓库可见性确实为 Public。

## 6. 发布第一个版本

代码验证完成后创建 `v0.1.0`：

```bash
git tag -a v0.1.0 -m "COBICount v0.1.0"
git push origin v0.1.0
```

在 GitHub 仓库右侧进入 `Releases`，选择 `Draft a new release`：

- Tag：`v0.1.0`
- Title：`COBICount v0.1.0`
- Release notes：概述严格 source-only 训练、独立目标域评估、环境和已知限制。

权重不要直接提交到普通 Git 历史。可将正式权重作为 Release asset 上传，并同时给出：

- 文件名；
- SHA-256；
- 使用的源域和划分；
- 训练命令；
- 对应 commit；
- PyTorch/CUDA 环境；
- 该权重是 EMA 还是 raw model。

## 7. 发布后最后检查

从一个新的空目录执行：

```bash
git clone https://github.com/YOUR_USERNAME/COBICount.git
cd COBICount
python -m venv .venv
# 激活环境
pip install -e ".[dev]"
pytest -q
cobicount-train --help
cobicount-evaluate --help
cobicount-infer --help
```

最后从 README 复制一条训练命令和一条评估命令进行实际验证。只有“新克隆环境”能够运行，才说明公开仓库不依赖你电脑中的隐式文件或路径。
