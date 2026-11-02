# Archive

Earlier versions of code that has since been replaced. Each folder is self-contained and still runs.

## v1_markov_chain

The original play-type Markov chain and its toy-data notebook, archived on 2026-10-14.

- `markovChain.py`: states are `(down, play_type, field_bucket)` from the *previous* play, with a 280 × 280 transition matrix that is marginalized down to the next play type.
- `conflatedPlayPrediction.ipynb`: trains it on simulated games and conflates it with a simulated classifier.

Replaced by the current `markovChain.py`, which predicts the next play type from the upcoming play's pre-snap situation (down, distance, field zone, team tendency, game situation) and is trained on real cfbfastR play-by-play data. On the 2017 season, v1 scored 54.3% accuracy / 0.870 log loss vs. 65.0% / 0.723 for the replacement (5-fold cross-validation by game).

To use v1 from code at the repo root:

```python
from archive.v1_markov_chain.markovChain import PlayMarkovChain
```
