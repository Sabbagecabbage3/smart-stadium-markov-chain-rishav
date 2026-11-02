# Setting up the Smart Stadium Markov chain

These steps get the play-type Markov chain running on your computer: about 10 minutes, plus a 72 MB data download.

## 1. Install the tools (skip any you already have)

- **Python 3.11 or newer:** https://www.python.org/downloads/ (on Windows, tick "Add python.exe to PATH" in the installer)
- **Git:** https://git-scm.com/downloads
- **VS Code** (recommended): https://code.visualstudio.com/, with the **Python** and **Jupyter** extensions

Check them in a terminal:

```bash
python --version
git --version
```

On macOS, use `python3` wherever these steps say `python`.

## 2. Get the code

```bash
git clone https://github.com/ehuang312/smart-stadium-markov-chain.git
cd smart-stadium-markov-chain
```

To push changes, first accept the collaborator invite from GitHub (in your email, or at https://github.com/ehuang312/smart-stadium-markov-chain/invitations).

## 3. Create a virtual environment and install packages

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. If PowerShell refuses to run `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

## 4. Check it works

```bash
python -m pytest tests -q
```

You should see `23 passed, 2 skipped`. The 2 skipped tests need the real data, which comes next.

## 5. Download the data and run the model

Start Python from the repo folder (`python`), then paste:

```python
from playByPlay import download_cfb_pbp, load_cfb_pbp
from markovChain import PlayMarkovChain

plays = load_cfb_pbp(download_cfb_pbp(2017))   # downloads once (~72 MB) into data/cfb_pbp/
mc = PlayMarkovChain().fit(plays)

situation = dict(offense='Georgia Tech', down=3, distance=8, yards_to_goal=65,
                 score_diff=0, half=1, period=2, seconds_left_half=600)
print(dict(zip(mc.play_types, mc.next_play_type_proba(situation).round(2))))
```

Expected output: about 27% run and 70% pass. Type `exit()` to leave Python. Running `python -m pytest tests -q` again should now show `25 passed`.

## 6. Open the notebook

1. In VS Code: **File → Open Folder** and pick `smart-stadium-markov-chain`. New terminals then start in the right folder with `.venv` active.
2. Open `notebooks/conflatedPlayPrediction.ipynb`.
3. Click **Select Kernel** (top right) and choose the `.venv` environment.
4. Click **Run All**. It takes about 3 minutes.

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'playByPlay'` | Python was started outside the repo folder. `cd` into `smart-stadium-markov-chain` first. |
| `No module named 'pandas'` (or numpy, pyarrow) | The virtual environment isn't active, or you're using a different Python (e.g. Anaconda's). Activate `.venv` (step 3). |
| The notebook can't find modules | The kernel isn't `.venv`. Use **Select Kernel** to switch. |
| Download fails | Check your internet connection; the data comes from github.com (SportsDataverse release). |

## Working in the repo

Read [README.md](README.md) for what the model does and [AGENTS.md](AGENTS.md) for the team rules. In short:

- Every model change becomes a new `versions/markovChain_vN.py` file.
- Never delete anything in `versions/` or `archive/`.
- Don't push without checking with the team.

If you use an AI coding agent (Claude Code, Codex, Cursor and others), it reads `AGENTS.md` automatically.
