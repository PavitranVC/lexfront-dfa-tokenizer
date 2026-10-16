# First prompt to paste into Claude Code

Read CLAUDE.md first. We're starting days 1-2: the honest baseline.

1. Write `experiments/data.py` with one loader per dataset (TweetTopic, GoEmotions
   simplified filtered to single-label comments, AG News). Check each dataset card for the
   recommended train/test split and tell me which split you chose and why.
2. Write `experiments/baseline.py`: tokenize with NLTK `word_tokenize`, TF-IDF, then
   (a) majority-class baseline and (b) Logistic Regression. Report accuracy and macro-F1,
   3 seeds, and save each run to `results/baseline/<timestamp>.json` with the full config.
3. Add pytest tests for the loaders (row counts > 0, labels in range, no empty texts).
4. Run it inside Docker and show me the results table, taken from the JSON files.

Explain each step briefly before you write the code, and keep functions small and readable.
