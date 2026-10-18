# Running a phase D notebook on Colab (instructions for whoever takes a dataset)

You need: a Google account with a little free Drive space, a browser, a stable connection. No GPU, no tokens, no coding.
Take **one dataset per person**: Civil Comments or Amazon polarity (estimate 4 to 7 hours each). Do not start Yelp until the project owner has decided on its grid (see the cost note in `docs/D_PROTOCOL.md`).

## Steps
1. Open the notebook from `notebooks/README.md` (the Colab badge in its row) and choose a **CPU** runtime.
2. In the second code cell set `QUICK_MODE = False` and leave `USE_DRIVE = True`. Do not change `DATASET`.
3. Run all cells (Runtime, Run all). Allow Google Drive access when asked. Lines starting `done` are new fits, lines starting `cached` were found from an earlier session.
4. Keep the laptop plugged in, awake, and the tab open. If Colab disconnects, open the **same notebook** again and run all cells again: finished fits are skipped.
5. When the last cells finish, a zip downloads (`lexfront_boost_<dataset>_results.zip`). Send it to the project owner; it goes into `results/boost_<dataset>/`.

## Rules that keep the results valid
* Do not edit the notebook or the repository between sessions: changed code makes saved fits count as stale and they are recomputed.
* Do not use a different dataset name or a different Drive folder when resuming.
* A run is only a result if `QUICK_MODE = False`; quick runs are dry runs.

## Troubleshooting
| Symptom | What to do |
|---|---|
| Disconnected or session ended | Reopen the same notebook and run all cells again. At most the fit that was running is lost. |
| "Out of memory" in the character baseline cell | Allowed. The report simply has no character row; carry on. |
| First cell says the clone could not be updated | Follow the message it prints (usually: Runtime, Disconnect and delete runtime, then run again). |
| Nothing prints for a long time | Normal during the wide search; each fit set can take tens of minutes. |

Review and merge steps (owner / assistant): run `python -m experiments.verify_boost --out results/boost_<dataset> --protocol docs/D_PROTOCOL.md --base origin/main`, check the generated `REPORT.md` against the protocol's hypotheses, then open a PR.
