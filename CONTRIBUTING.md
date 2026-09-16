# Contributing

感谢你愿意改进 MiniGPT。这个项目优先接受可读性、可复现性和教学价值较高的改动。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 提交前检查

```bash
python scripts/sanity_check.py
python -m pytest
python -m ruff check .
```

## 改动原则

- 保持模型实现简洁，避免引入与学习目标无关的抽象。
- 修复 Bug 时尽量补充对应测试。
- 改变训练参数、数据处理或解码策略时，在 `EXPERIMENT_LOG.md` 中记录实验条件。
- 不要提交 `__pycache__`、虚拟环境、临时输出或本地日志。
- 提交信息建议使用 Conventional Commits，例如 `fix: validate top-p range`。

## Pull Request

Pull Request 中请说明：

1. 改动解决了什么问题。
2. 是否改变模型行为或 Checkpoint 兼容性。
3. 运行了哪些检查，以及结果如何。
4. 如需实验数据，请附上可复现命令和关键指标。
