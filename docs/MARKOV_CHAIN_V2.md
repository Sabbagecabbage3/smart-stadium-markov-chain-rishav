# Play-Type Markov Chain v2

**Smart Stadium VIP: play-type prediction, October 2026**

The Markov chain is the context half of the play-type predictor (paper Sections 4.2–4.3). It predicts the type of the upcoming play from the game situation, and its probabilities are conflated with a video classifier's. v2 replaces the original toy-data version with a model trained on real NCAA play-by-play data. It predicts from the upcoming play's pre-snap situation instead of the previous play.

**Headline results** (2017 FBS season, 5-fold cross-validation, 140,279 plays):

| | v1 | v2 |
|---|---|---|
| Markov chain alone, all plays | 54.3% | **64.8%** |
| Markov chain alone, Georgia Tech offense | 67.3% | **80.3%** |
| Combined with a simulated ~71% classifier, all plays | 84.8% | **87.3%** |
| Combined with a simulated ~71% classifier, Georgia Tech | 84.3% | **90.1%** |

The combined rows use a single 80/20 split by game (section 4.3). The classifier is still simulated, so treat those rows as indicative until real CNN output exists.

---

## 1. What's in this package

| File | What it is |
|---|---|
| `markovChain.py` | **Rewritten.** The v2 model (`PlayMarkovChain`), its markers and `team_run_rates`. |
| `playByPlay.py` | **New.** Downloads and cleans cfbfastR play-by-play data into one row per play. |
| `conflation.py` | Unchanged. Fuses the Markov chain with classifier probabilities. Included because the notebook uses it. |
| `notebooks/conflatedPlayPrediction.ipynb` | **Rewritten.** Trains on real data, compares against v1, runs conflation and the marker cross-validation. |
| `tests/test_markov_chain.py`, `tests/test_play_by_play.py` | **New.** 25 unit tests. |
| `archive/v1_markov_chain/` | The original `markovChain.py` and notebook. The notebook only gained an "archived" note and an import that points at the archived code, so v1 still runs. |
| `requirements.txt` | Packages needed for this part of the project only. |
| `docs/MARKOV_CHAIN_V2.md` | This document. |

This document was first shipped in `markov_chain_v2.zip`; the same files now live in this repo, with each model version also saved in `versions/`. The play-by-play data isn't included. It downloads automatically (about 72 MB per season) into `data/cfb_pbp/`.

## 2. Quick start

```bash
pip install -r requirements.txt
```

```python
from playByPlay import download_cfb_pbp, load_cfb_pbp
from markovChain import PlayMarkovChain

plays = load_cfb_pbp(download_cfb_pbp(2017))      # downloads once into data/cfb_pbp/
mc = PlayMarkovChain().fit(plays)

situation = dict(offense='Georgia Tech', down=3, distance=8, yards_to_goal=65,
                 score_diff=0, half=1, period=2, seconds_left_half=600)
mc.next_play_type_proba(situation)   # -> [0.27, 0.70, 0.00, 0.00, 0.03] over ('R', 'P', 'F', 'U', 'O')
mc.describe(situation)               # -> {'down': '3rd', 'distance': '7-10', 'field_zone': 'own side',
                                     #     'team_tendency': 'run-heavy', 'game_situation': 'normal'}
```

- **Notebook:** open `notebooks/conflatedPlayPrediction.ipynb` and Run All. It takes about 3 minutes, mostly section 6.
- **Tests:** `python -m pytest tests -q`. The two real-data tests skip themselves if the 2017 file hasn't been downloaded.

## 3. What changed from v1

### 3.1 Design

| | v1 | v2 |
|---|---|---|
| **Predicts from** | The *previous* play's state | The *upcoming* play's pre-snap situation |
| **Inputs** | Down, play type, field position (10-yard buckets) | Down, distance to go, field zone, offense run tendency, game situation (score + clock) |
| **Model** | 280 × 280 state-transition matrix (78,400 cells), summed down to play type | Play-type counts for 1,200 contexts × 5 play types (6,000 cells) |
| **Rare situations** | Add 0.001 to every transition (nearly uniform when unseen) | Back off to the next-coarser context |
| **Training data** | Simulated games | Real NCAA play-by-play (2017: 869 games, 149,883 plays) |
| **Play types predicted** | 7: run, pass, PAT, field goal, punt, kickoff, other | 5: run, pass, field goal, punt, other, matching the classifier's classes |
| **Configurable** | No | Markers, bucket edges, back-off strength, run-rate source |
| **Tests** | None | 25 |

**Why the switch to pre-snap situation:** play calls are driven by the situation the offense faces *now*, such as 3rd & 8 or 4th & 2. That's all visible on the scoreboard before the snap. v1 saw only the previous play, so it had to guess the current down and distance indirectly. On real data, a model using only down, distance and field zone (100 contexts) already beats v1 by a wide margin (section 4.1).

**Why the state space stayed small:** v1 spent most of its matrix predicting the next down and field position, then threw that away. v2 predicts only play type, which frees enough room for two new markers while still using fewer cells than v1.

### 3.2 Breaking API change

`next_play_type_proba()` now takes the *upcoming* play's situation as a dict instead of the previous play's `(down, play_type, field_bucket)` tuple. `fit()` now takes plays instead of game sequences.

```python
# v1
mc = PlayMarkovChain()
mc.fit(play_sequences)                     # list of games, each a list of (down, play_type, field_bucket)
mc.next_play_type_proba((1, 'R', 5))       # state of the play that just happened

# v2
mc = PlayMarkovChain().fit(plays)          # DataFrame or list of dicts, as returned by load_cfb_pbp
mc.next_play_type_proba(situation)         # pre-snap situation of the play about to happen
```

**Removed:** `DOWNS`, `FIELD_BUCKETS`, `states`, `state_to_idx`, `transition_matrix` and `next_state_proba()`. The PAT (`X`) and kickoff (`K`) codes are no longer predicted. Nothing else in the repo used these. v1 is still importable with `from archive.v1_markov_chain.markovChain import PlayMarkovChain`.

## 4. Prediction accuracy

### How it was measured

- **Data:** 2017 FBS season, all scrimmage plays (kickoffs excluded): 140,279 plays.
- **Cross-validation:** 5 folds split by game, so a test game is never seen in training. Every game is tested exactly once, and team run rates come only from training games.
- **Two metrics.** *Accuracy* is how often the top-ranked play type is right. *Log loss* scores the whole probability vector (lower is better), which matters because conflation uses the full probabilities, not just the top guess.
- **Georgia Tech** is reported separately (844 plays) because the project's footage is GT football and GT's option offense ran on 79–81% of plays. It's a small sample, so expect a few points of noise.

### 4.1 Markov chain on its own

| Model | Contexts | Log loss | Accuracy | GT log loss | GT accuracy |
|---|---|---|---|---|---|
| Overall play-type rates only (baseline) | 1 | 1.077 | 46.1% | 1.021 | 72.7% |
| **v1 (archived)** | 280 states | 0.870 | 54.3% | 0.818 | 67.3% |
| Down + distance + field zone | 100 | 0.753 | 61.9% | 0.714 | 71.7% |
| … + team tendency | 300 | 0.744 | 62.9% | 0.588 | 79.5% |
| **… + game situation (v2 default)** | 1,200 | **0.723** | **64.8%** | **0.568** | **80.3%** |
| … + previous play type (optional) | 8,400 | 0.722 | 65.3% | 0.566 | 80.1% |

- **v2 vs v1:** +10.5 accuracy points overall and +13.0 for Georgia Tech, with log loss down 17% and 31%.
- **v1 on GT:** v1 does *worse* than always predicting the overall rates (67.3% vs 72.7%), because it has no notion of team tendency.
- **Top-2 accuracy:** the right play type is in the top two guesses 96.5% of the time with v2, vs 92.8% with v1.

### 4.2 Where the gains come from

Recall is the share of each play type correctly identified. Precision is how often a prediction of that type is right.

| Play type | Plays | v1 recall | v2 recall | v1 precision | v2 precision |
|---|---|---|---|---|---|
| Run | 64,632 | 69.5% | 70.4% | 54.2% | 63.3% |
| Pass | 59,831 | 36.6% | **57.6%** | 55.8% | 62.8% |
| Field goal | 2,642 | 61.2% | **82.6%** | 52.7% | 70.1% |
| Punt | 8,911 | 87.0% | **98.2%** | 51.3% | **85.0%** |
| Other | 4,263 | 0.0% | 1.1% | n/a | 97.9% |

| Down | Plays | v1 accuracy | v2 accuracy |
|---|---|---|---|
| 1st | 56,421 | 48.3% | 60.7% |
| 2nd | 42,026 | 53.0% | 60.6% |
| 3rd | 26,991 | 60.2% | 72.1% |
| 4th | 14,841 | 70.1% | 79.4% |

- **Passes:** v2's distance marker lets it recognize passing downs (3rd & long), which v1 mostly called runs.
- **Kicks:** punts and field goals become much more reliable once field zone and distance are known together.
- **"Other"** is rarely the most likely type in any situation (pre-snap fouls are scattered across all of them), so the Markov chain almost never predicts it. It still assigns it a small probability, which is what conflation needs.

### 4.3 Combined with a classifier (conflation)

No real video classifier output exists yet. The notebook simulates one at about 71% accuracy, close to the paper's CNN at 75.3%. This uses a single 80/20 split by game: 27,941 test plays, 242 by Georgia Tech.

| Prediction | All plays | Georgia Tech |
|---|---|---|
| Simulated classifier alone | 71.0% | 69.8% |
| v1 Markov chain alone | 54.6% | 67.8% |
| v2 Markov chain alone | 65.1% | 80.6% |
| Classifier + v1 | 84.8% | 84.3% |
| **Classifier + v2** | **87.3%** | **90.1%** |
| Classifier + v2 with learned class weights | 87.3% | 89.7% |

- **Overall:** swapping v1 for v2 adds 2.5 points after conflation.
- **Georgia Tech:** it adds 5.8 points.
- **Learned weights:** the class weights from `fit_conflation_weights` don't add anything on top of plain conflation here.
- **For comparison:** the paper reports 75.3% for its classifier alone and 87.2% conflated.

**About v1's old numbers:** the v1 notebook reported 50.5% for the Markov chain alone and 92.6% conflated. Those came from *simulated* games and a stronger simulated classifier (88.5%), so they aren't comparable with anything above. On real data, v1 alone scores 54.3%.

### 4.4 Why these markers and buckets

Each row changes one thing from the v2 default (log loss 0.723, accuracy 64.8%):

| Change | Contexts | Log loss | Accuracy | GT accuracy |
|---|---|---|---|---|
| Drop down | 300 | 0.969 | 54.5% | 72.6% |
| Drop field zone | 240 | 0.769 | 62.8% | 78.9% |
| Drop distance | 240 | 0.756 | 61.9% | 77.1% |
| Drop game situation | 300 | 0.744 | 62.9% | 79.5% |
| Drop team tendency | 400 | 0.728 | 64.2% | 71.8% |
| 3 distance buckets instead of 5 | 720 | 0.728 | 64.4% | 80.1% |
| No "end of 1st half" game situation | 900 | 0.728 | 64.4% | 80.3% |
| Score and clock as two separate markers | 1,800 | 0.732 | 64.0% | 80.0% |
| 10 field buckets instead of 5 | 2,400 | 0.726 | 64.7% | 80.5% |

- **Dropping a marker:** every one costs accuracy. Down matters by far the most.
- **Team tendency:** barely matters overall but is worth 8.5 points for Georgia Tech.
- **Alternative buckets:** none beat the defaults on log loss, whether they use fewer contexts or more.
- **Back-off strength:** results are flat for k between about 5 and 50, and the default is 25.
- **Coverage:** with one season, 98% of training plays fall in a context with at least 10 plays, so the data supports the state space.

## 5. New features

### 5.1 Real data loader (`playByPlay.py`)

- **`download_cfb_pbp(season)`** fetches one season from the SportsDataverse GitHub release (tag `cfbfastR_cfb_pbp`, 2014–2026, no API key). It skips the download if the file already exists.
- **`load_cfb_pbp(paths)`** takes one file or a list of seasons. It returns one row per play, in game order, with the pre-snap fields the model uses plus `play_type`, `prev_play_type` and `raw_play_type`.
- **`map_play_types()`** collapses cfbfastR's ~40 labels into the model's codes:
  - Sacks and interceptions count as passes. Fumbles use the data's rush/pass flags.
  - About 7,600 **"Penalty" rows** are sorted by their text. A run or pass wiped out by a flag keeps its type, and pre-snap fouls with no play become **other (O)**. Those fouls make up almost all of the O class.
  - **PATs** aren't separate rows in this data, so there is no PAT code.
  - **Timeout, end-of-period and end-of-game** rows are dropped.
  - **Bad values:** 0 or negative distances are set to 1.

### 5.2 Markers

A marker turns raw pre-snap fields into a few buckets. The defaults, in back-off order:

| Marker | Buckets |
|---|---|
| `down` | 1st, 2nd, 3rd, 4th |
| `distance` | 1, 2–3, 4–6, 7–10, 11+ yards |
| `field_zone` | red zone (≤ 20 to go), opp 21–35, opp 36–49, own side, own deep (inside own 20) |
| `team_tendency` | Offense's run rate: pass-heavy (< 45%), balanced, run-heavy (≥ 60%). Unknown teams count as balanced. |
| `game_situation` | normal, trailing late, leading late (last 8 min of the 4th quarter), end of 1st half (last 2 min) |
| `prev_play_type` *(optional)* | previous play's type, or start of game |

- **Choosing markers:** `PlayMarkovChain(markers=(...))` takes any subset or order.
- **Adding a marker:** add an entry to `MARKERS`. Section 6 of the notebook shows how.

### 5.3 Back-off smoothing

Predictions start from the overall play-type rates and move toward each finer context in turn:

p = (counts + k · p_coarser) / (total + k), with k = `backoff_strength` (default 25)

A situation with thousands of plays uses its own rates. A rare one, such as 4th & 1 in the red zone while trailing late, leans on "4th & short in the red zone". This is what allows more markers without needing more data.

### 5.4 Team tendency and avoiding leakage

- **Default:** `fit()` computes each offense's run rate from the training plays.
- **Predicting a live season:** pass last season's rates instead, so the model never uses the season it's predicting:

```python
mc = PlayMarkovChain().fit(plays, run_rates=team_run_rates(last_season_plays))
```

### 5.5 Other additions

- **`predict_proba(plays)`** predicts many plays at once and computes each distinct context only once. A full fit plus predictions on one season takes about 1 second.
- **`describe(play)`** gives a readable context, and **`n_contexts`** gives the size of the context space.
- **Inputs:** `fit()` and `predict_proba()` accept a DataFrame or a list of dicts.

## 6. Limitations and next steps

- **The classifier is simulated.** The conflated numbers will change with real CNN probabilities. Rerun notebook section 5 once they exist.
- **The meaning of "other" needs agreement with the video team.** In this data it means pre-snap fouls. If the video labels use "other" for ads or replays, the Markov chain shouldn't be asked to predict it.
- **One season of training data.** `load_cfb_pbp` accepts a list of seasons. Adding 2014–2016 is the next step, and 2016 matches some of the project's video clips.
- **Team tendency is from the same season** in the reported results, using only training games. A live deployment should use the previous season's rates (section 5.4).
- **Kickoffs aren't predicted.** Their down and distance fields are placeholders, and the classifier has no kickoff class either.
- **Georgia Tech numbers rest on 242–844 plays**, so they carry more noise than the overall numbers.
