---
title: "LexFront Review 1: Screenshot Guide and Finalization Guide"
subtitle: "What to capture, how to run it, and how to make the report your own"
---

# 0. Read this first

This guide has three parts.

1. **Part A:** every screenshot you might paste into the report, how to produce it, and where it goes.
2. **Part B:** how to turn the draft into your own report, by hand, efficiently.
3. **Part C:** how to produce the graphs and diagrams yourselves instead of using the ones in the draft.

Three ground rules apply throughout.

- **Numbers never get retyped.** Every number in the report comes from `results/`. If you edit prose, leave the numbers alone. If a number must change, change the results file by re-running the code, then rebuild.
- **A screenshot must show something you actually ran.** Do not mock one up or edit pixels. If a screenshot shows different numbers from the report, say why in its caption (see A.2).
- **Follow your course's rule on AI assistance.** The draft was written with AI help. Check what your course allows, and disclose it the way your course asks. Part B is about making sure the report is accurate, understood and in your own words. It is not about hiding where a first draft came from, and nothing here is meant to get past an AI detector.

# Part A: Screenshots

## A.1 One-time setup

You need these on the machine where you take screenshots. Do this once.

1. Install Python 3.11 and Git.
2. Clone the repo and switch to the report branch:
   ```
   git clone https://github.com/PavitranVC/lexfront-dfa-tokenizer.git
   cd lexfront-dfa-tokenizer
   git checkout claude/upbeat-feynman-itf5yi
   ```
3. Create a clean environment and install the packages:
   ```
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords')]"
   ```
   The NLTK download is required. Without it the tests that use NLTK fail with a `LookupError`.
4. For the DFA picture, install Graphviz so the `dot` command exists (Ubuntu: `sudo apt install graphviz`; Mac: `brew install graphviz`; Windows: installer from graphviz.org, then tick "add to PATH"). Check with `dot -V`.
5. Screenshot tools: Windows `Win+Shift+S`, Mac `Cmd+Shift+4`, Linux `PrtSc` or Flameshot. Set the terminal font to a readable size (about 14 pt) and use a light theme so it prints well.

Before every screenshot: crop to the relevant part, and make sure no email address, token or personal folder name is visible.

## A.2 Important: the notebook has a quick mode with different numbers

The Colab notebook starts with `QUICK_MODE = True`. That uses a small fixed subsample (2,000 train and 500 test documents per dataset) and writes to `results/quick/`. Its numbers are **not** the report's numbers.

You have two safe options for notebook screenshots:

- **Option 1 (recommended):** run the notebook in quick mode, and for any screenshot that shows accuracy or macro-F1, use the last cell, "Appendix, committed full-data results". It prints the full-data tables from `results/`, which match the report.
- **Option 2:** set `QUICK_MODE = False` and run everything. This reproduces the full results but is slow, because AG News has a large training set. Use it only if you have time.

If you do paste a quick-mode screenshot, put this in its caption: "Quick mode (fixed-seed subsample); numbers differ from Tables 4 to 9, which use the full data."

## A.3 The screenshot list

IDs are used below. "Where" refers to sections of the current report.

| ID | What to capture | Where it goes |
|:---|:----------------------------|:---------------------|
| S1 | Repository file tree | Section 3 or Appendix |
| S2 | `pytest -q` passing | Section 3.2 or Appendix |
| S3 | GitHub Actions run, green | Appendix |
| S4 | Console output of a Module 1 run | Section 4.2 or Appendix |
| S5 | One result JSON file opened | Section 3.1 or Appendix |
| S6 | Notebook: datasets table | Section 4.1 |
| S7 | Notebook: Module 1 table and bar chart | Section 4.2 |
| S8 | Notebook: side-by-side tokenizer table | Section 4.3 |
| S9 | Notebook: LexFront token table for one tweet, and symbol table | Section 3.2 or 4.3 |
| S10 | Notebook: DFA state table and DFA picture | Section 4.3 |
| S11 | Notebook: scanner demo with UNKNOWN and the agreement line | Section 4.3 |
| S12 | Notebook: intrinsic metrics table and chart | Section 4.3 |
| S13 | Notebook: existing vs proposed table and chart | Section 5.1 |
| S14 | CHANGELOG and `git log` | Appendix |

You do not need all of them. The report is complete without any screenshots. Use S2, S5, S10, S11 and S13 if you want to keep it short, because they show testing, reproducibility, the DFA and the main result.

## A.4 How to produce each one

### S1: Repository file tree

1. In the repo folder run `ls` (or `tree -L 2 -I '.venv|.git'` if `tree` is installed).
2. Capture the output. It should show `lexer/`, `experiments/`, `tests/`, `results/`, `notebooks/`, `docs/`, `report/`.

### S2: Tests passing

1. Activate the environment and run:
   ```
   pytest -q
   ```
2. Wait for the final line, for example "N passed in X s". Capture the whole terminal, with that last line visible.
3. The changelog records 99 tests from a clean clone. Your own count is what to report. If any test fails, fix the cause first (usually the missing NLTK data from A.1) and do not paste a failing run.

### S3: CI green on GitHub

1. Open the repo on GitHub, click the **Actions** tab.
2. Open the latest run on `main` and capture the page showing both jobs, `tests` and `notebook`, with green ticks.

### S4: Console output of a Module 1 run

The committed results already exist, so a normal run skips them and prints nothing useful. Write to a scratch folder instead, so the committed `results/` is not touched:

1. Run:
   ```
   python -m experiments.run_module1 --datasets ag_news --max-train 2000 --max-test 500 --out results/scratch_demo
   ```
2. You will see one line per run, such as the dataset, the tokenizer, the classifier and its accuracy and macro-F1. Capture them.
3. Caption it as a demonstration on a subsample. Afterwards delete the scratch folder (`rm -r results/scratch_demo`) and do not commit it.

### S5: A result JSON file

1. Open `results/module1/tweet_topic__nltk__nb.json` in any editor.
2. Capture the top part. It shows the dataset and split, vectorizer settings, classifier, seed, metrics and library versions. This is your evidence that each run is recorded with its full configuration.

### S6 to S13: Notebook screenshots

1. Open the notebook from the Colab badge at the top of `README.md`, or upload `notebooks/LexFront_Review1.ipynb` to Colab.
2. Choose **Runtime, Run all**. The first cell installs the packages and downloads the NLTK data. If Colab asks you to restart the runtime, do so and run all again.
3. Wait for it to finish. Quick mode takes a few minutes on a CPU runtime.
4. Scroll to each section and capture the output cell, using the table below. Remember A.2 for anything that shows accuracy or F1.

| ID | Notebook section | What to capture |
|:---|:--------------------|:------------------------------------|
| S6 | 1. Datasets | The datasets table |
| S7 | 2. Module 1 | The results table and the accuracy chart |
| S8 | 3a. Side-by-side | The table of four tokenizers on eight tweets |
| S9 | 3a and 3b | One per-tweet table (class, lexeme, normalized) and the symbol table |
| S10 | 3c. Own DFA engine | The state-count table and the DFA picture |
| S11 | 3d. Scanner demo | The token table (with the `UNKNOWN` row for `§`) and the line "DFA scanner and PLY lexer agree on this input: True" |
| S12 | 3e. Intrinsic metrics | The table and the three-panel chart |
| S13 | 4. Existing vs proposed | The table and the two-panel chart |

If a figure in the report and a notebook screenshot show the same thing, use one or the other, not both.

### S14: CHANGELOG and git log

1. Run `git log --oneline -15` and capture it.
2. Open the top of `CHANGELOG.md` and capture the newest entries. This shows the design history.

## A.5 How to put a screenshot into the report

1. Save each image into `report/figures/` with a clear name, for example `shot_pytest.png`. Use PNG.
2. Open `report/Review1_Report.template.md` and add a block at the end, before the `# References` heading, so you do not disturb the existing figure numbers:
   ```
   # Appendix A. Implementation evidence

   ![Figure A1. All tests passing (`pytest -q`).](figures/shot_pytest.png){width=14cm}
   ```
3. Write a caption that says what the image shows and where it came from, and add one sentence of your own on what it proves.
4. Rebuild (Part B, step 6).

# Part B: Making the report your own, step by step

The goal is a report that is accurate, that you understand line by line, and that reads like you wrote it. You will also be asked about it in the viva. Plan for about half a day with three people.

## B.1 Where the text lives

- **Edit this file:** `report/Review1_Report.template.md`. It holds all the prose.
- **Do not edit by hand:** `report/Review1_Report.md` and `report/Review1_Report.docx`. They are generated from the template and are overwritten on every rebuild.
- `report/refs.py` holds the reference list. `report/build_report.py` inserts the tables from `results/tables/` and numbers the citations.
- In the template, `{{key}}` is a citation, and `@@TABLE:...@@` inserts a results table. Leave those markers alone.

## B.2 Split the work

Divide the report among the three of you. A simple split:

- Person A: Abstract, Section 1 (introduction, version history) and Section 6.
- Person B: Section 2 (literature survey) and the references.
- Person C: Sections 3, 4 and 5 (the system, results and metrics).

Each person edits only their own sections, so there are no merge conflicts, and each person must be able to explain their sections out loud.

## B.3 The edit loop for each section

Do this for every section you own. It is the most efficient way to get real understanding and your own wording at the same time.

1. **Read** the draft section once, slowly.
2. **Close it.** Write 3 to 5 bullet points in your own words: what was done, why, and what the result was. Use your own memory of the project, and look at the code if you are unsure.
3. **Write the paragraph again** from your bullets, in the way you would explain it to a classmate. Keep sentences short and plain.
4. **Compare with the draft.** Check that you did not drop a fact, and that you did not add a claim the data does not support.
5. **Put back every number** exactly as in the table, by copying from `results/tables/` rather than typing from memory.
6. Move on. Do not polish more than once per section.

## B.4 What to change, and what not to change

Change these:

- **Generic sentences.** Replace them with what actually happened in your project. Real events you can use are in `CHANGELOG.md` and `docs/`: the TweetTopic loading problem and the Parquet workaround, the Hugging Face outage, why you chose the PLY dispatcher, why Hinglish and Docker were dropped, the ELONGATED back-reference finding.
- **Explanations you cannot give from memory.** If you cannot explain a sentence, either read the code or `docs/LEARN.md` until you can, or delete the sentence.
- **Wording.** Use the vocabulary you normally use, keep academic tone, and keep paragraphs short.

Do not change these:

- **The numbers, tables and the conclusions drawn from them.** The honest statements in Section 5.4 (the gain comes mainly from the classifier, results are single runs, TF-IDF lowercasing makes the lowercase rule redundant) must stay.
- **References.** Do not add one you have not opened. Do not remove the `[TEAM]` or `[MISSING]` markers until you have actually filled them (B.5).

## B.5 Fill the open markers

Search the template for `[TEAM` and `[MISSING`. There are three items (four markers, because the textbook one appears twice):

1. **Base-paper citation** (Section 1.2). Add the real paper: title, authors, venue, year and URL. Add it as a new entry in `report/refs.py` and cite it with `{{key}}`. Open the link yourself before adding it.
2. **Course textbook citation** (Sections 2.3 and 3.2). Add the textbook you used for subset construction and minimization, in IEEE style.
3. **Reason for dropping Docker** (Section 1.4, version table). Write the real reason, or delete the sentence.

When a marker is filled, delete its bracket text.

## B.6 Rebuild and check

After any edit:

```
pip install python-docx          # once
python report/build_report.py --docx
```

This needs `pandoc` installed (`pandoc --version` to check). It rewrites the `.md` and `.docx`. Then:

1. Open the `.docx` in Word. Check that every table and figure is present and every caption is numbered.
2. Search the document for `[TEAM`, `[MISSING` and `{{`. None should remain at the end (a leftover `{{` means a mistyped citation key).
3. Check that page numbers show in the footer.

## B.7 Final polish in Word, by hand

Do this once, after the last rebuild, because a rebuild overwrites changes made in Word.

1. Read the whole document aloud, or have a teammate read it to you. Fix anything awkward.
2. Check headings, figure and table captions, and that no table is split badly across a page.
3. Add your institution, course code and roll numbers to the title block if your department requires it.
4. Export a PDF (File, Save As, PDF) and check it once more before submitting.

## B.8 Check the claims one last time

Before submitting, confirm each of these:

- Every number in the text matches a table in `results/tables/`.
- Every citation number points to the right reference, and you have opened each link at least once.
- No sentence calls LexFront "first" or "novel", and none claims the tokenizer caused the accuracy gain.
- You can explain every figure and every table in about a minute.

For viva preparation, use the 30 questions in `docs/LEARN.md` as a self-test.

# Part C: Producing the graphs and diagrams yourselves

The figures in the draft were generated from the project's own code and result files. Using figures you generate yourselves is good, as long as the data behind them is the real data. Choose one option per figure.

## C.1 Bar charts (Figures 5, 7, 8 and 9)

### Option 1: Regenerate with the project code (easiest)

1. Activate the environment from A.1.
2. Run:
   ```
   python -c "from experiments.figures import build_all_figures; print(build_all_figures('results'))"
   ```
3. The charts are rewritten into `results/figures/`. Copy them over the ones in `report/figures/` (same file names), then rebuild.
4. If you want a different look (colors, fonts, sizes), edit `experiments/figures.py` and run it again. The data comes from the JSON, so the numbers stay correct.

### Option 2: Plot them yourself in a spreadsheet or plotting tool

1. Open `results/tables/module1_results.md` (or the table for the figure you want).
2. Copy the numbers into Excel, Google Sheets or your own matplotlib script, by copy and paste from the table file. Do not retype them.
3. Make a grouped bar chart: datasets on the x-axis, one bar per tokenizer (or per setting), accuracy on the y-axis.
4. Label every bar with its value, and then check each label against the table.
5. Export as PNG at 200 dpi or more, about 16 cm wide, and save it in `report/figures/` using the same file name as the one it replaces.

## C.2 DFA diagram (Figure 6)

- **Regenerate:** with Graphviz installed, run
  ```
  dot -Tpng results/module2/dfa_min.dot -o report/figures/dfa_min.png
  ```
- **Or redraw by hand** in diagrams.net (draw.io) or on paper and photograph it. Keep the same states `q0` to `q9`, the same transitions and the same accepting states (`q5` WORD, `q6` HASHTAG, `q8` EMOTICON, `q9` MENTION), and compare it with the committed picture when you are done. A hand drawing is allowed to look different, but it must say the same thing.

## C.3 Architecture diagrams (Figures 1 to 4)

The source of each is a short text file in `report/diagrams/` (`v1.mmd`, `v2.mmd`, `v3.mmd`, `current.mmd`).

- **Redraw by hand** in diagrams.net, PowerPoint or on paper. Follow the boxes and arrows in the source file, and keep the content the same (for example, v3 must still show "LexFront + BPE" as later work).
- **Or re-render the Mermaid:** paste the text of a `.mmd` file into https://mermaid.live, then export a PNG. Or run `npx @mermaid-js/mermaid-cli -i report/diagrams/v1.mmd -o report/figures/arch_v1.png`.

## C.4 Rules for any figure you replace

1. Keep the same file name in `report/figures/`, so the report picks it up on the next rebuild.
2. Keep the caption honest. For a redrawn diagram say "Redrawn from `docs/versions/...`". For a chart say that it is built from the JSON files in `results/`.
3. Use a resolution that stays sharp at 16 cm wide (200 dpi or more), readable fonts, and axis labels and units.
4. Re-check every value against the matching table.
5. In Word, you can swap a picture by right-clicking it and choosing **Change Picture**. But a rebuild undoes this, so do it only at the final polish step (B.7).

# Quick checklist

- [ ] Environment set up, NLTK data downloaded, `dot -V` works
- [ ] `pytest -q` passes (S2)
- [ ] Screenshots taken, cropped, personal details removed, quick-mode ones captioned
- [ ] Each section rewritten with the B.3 loop, numbers untouched
- [ ] Three open items (four markers) filled or removed
- [ ] `python report/build_report.py --docx` run, `.docx` checked
- [ ] No `[TEAM`, `[MISSING` or `{{` left
- [ ] Course AI-assistance disclosure added in the form your course requires
- [ ] PDF exported and read once more
