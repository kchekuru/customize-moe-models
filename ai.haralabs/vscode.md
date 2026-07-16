# Visual Studio Code setup

This repo is set up to open cleanly in VS Code.

## One-time setup

1. **Clone and open**
   ```bash
   git clone https://github.com/kchekuru/customize-moe-models.git
   code customize-moe-models
   ```
2. When VS Code prompts, **install recommended extensions** (Python, Pylance, Ruff, debugpy).
3. Create the env and install (from the repo root):
   ```powershell
   uv venv .venv
   uv pip install -e ".[dev,inkling]" --python .\.venv\Scripts\python.exe
   ```
4. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
5. Set your API key (User settings, or a local terminal / `.env` you do not commit):
   ```powershell
   $env:TINKER_API_KEY = "your-key"
   ```

## What is already configured

| File | Purpose |
|------|---------|
| `.vscode/settings.json` | Interpreter, pytest, Ruff format-on-save, PYTHONPATH |
| `.vscode/extensions.json` | Recommended extensions |
| `.vscode/launch.json` | Debug configs for Inkling SFT/RL/Code RL + pytest |

## Run / debug

- **Run and Debug** panel (`Ctrl+Shift+D`):
  - `SL basic (Inkling)` — supervised fine-tune starter
  - `RL basic (Inkling)` — RL starter
  - `Code RL (Inkling)` — coding RL (needs sandbox if grading runs code)
  - `Pytest unit tests` / `Pytest current file`
- **Testing** sidebar: discovers tests under `tinker_cookbook/` and `tests/`.
- Integrated terminal auto-activates `.venv`.

## Inkling model id

Use `thinkingmachines/Inkling` anywhere a recipe accepts `model_name`. Renderer/tokenizer are selected automatically when the `inkling` extra is installed.
