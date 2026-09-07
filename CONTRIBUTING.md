# Contributing to HanBayes

感谢你对 HanBayes 的关注！欢迎任何形式的贡献：bug 报告、文档改进、新功能、性能优化。

Thanks for your interest in improving HanBayes! Bug reports, documentation
improvements, new features and performance work are all welcome.

> 本项目交流以英文或简体中文均可。You may use English or Simplified Chinese.

## 开发环境 / Development setup

```bash
git clone https://github.com/GOOD-123-CPU/hanbayes.git
cd hanbayes
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev,plot]
```

验证环境 / Verify:

```bash
pytest tests/ -v          # all tests must pass
ruff check src/ tests/    # lint must be clean
```

## 流程 / Workflow

1. Fork 仓库并从 `main` 拉出特性分支（`feat/xxx` 或 `fix/xxx`）；
2. 保持每个 PR 聚焦单一改动；
3. 提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 风格
   （`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`）；
4. 确保测试与 lint 全部通过——CI 会在 Python 3.9–3.13 × Linux/macOS/Windows 上运行；
5. 新功能必须附带单元测试；bug 修复必须先添加能复现该 bug 的测试。

## 代码规范 / Code style

- 使用 ruff 进行 lint（配置见 `pyproject.toml`）；
- 类型注解：公开 API 必须有完整类型标注（包已带 `py.typed`）；
- Docstring：公开 API 使用英文 docstring（面向国际用户），算法说明可辅以中文；
- **兼容性红线**：任何改动都不得破坏论文冻结数字的精确复现
  （`hanbayes final-test` 输出必须与 `results/` 完全一致）。

## 报告 bug / Reporting bugs

提交 issue 时请包含：

- Python 版本与操作系统；
- 最小可复现代码（不要粘贴数据集内容，ChnSentiCorp 受其自身许可约束）；
- 实际行为 vs 期望行为（含完整 traceback）。

## 提交功能建议 / Feature requests

请先搜索现有 issue；新建时说明使用场景与预期 API 形态。

## 行为准则 / Code of conduct

参与本项目即表示你同意遵守
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
——保持友善、尊重与专业。
