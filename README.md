# Fake or Real — The Impostor Hunt in Texts

CENG313, Group 66. Assignment 1: exploratory data analysis and baseline models.
## İbrahim Emir Tosun 23118080074


Competition: https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt

## Dataset and approach

Each article contains `file_1.txt` and `file_2.txt`; predict which text is real.
There are 95 labelled training pairs and 1,068 test pairs. EDA examines 93
non-empty pairs. Model evaluation retains all 95 pairs, including empty texts.

Baselines: majority position, shorter text, and TF-IDF + Logistic Regression.
Pairs remain intact in the 80/20 split and five-fold stratified cross-validation.
TF-IDF is fitted on training texts only. Validation results are exploratory
because EDA preceded the split; shared partial content may also affect them.

## Run locally

1. Create and activate a Python 3.12 virtual environment.
2. Run `python -m pip install -r requirements.txt` from the project root.
3. Download competition data through Kaggle after accepting its rules.
4. Extract it so `data/raw/train.csv`, `data/raw/train/` and `data/raw/test/` exist.
5. Open `notebooks/assignment1_eda_baseline.ipynb`, select the environment,
   and run all cells from top to bottom.

## Run in Google Colab

Open a Colab notebook and run these setup cells first:

```python
!git clone https://github.com/tosuun/GROUP66_CENG313.git /content/GROUP66_CENG313
%cd /content/GROUP66_CENG313
!pip install -r requirements.txt
```

Upload or copy the downloaded competition data into
`/content/GROUP66_CENG313/data/raw/`, using the same layout as above.
Then copy the assignment notebook into this Colab session (File > Upload
notebook), rerun `%cd /content/GROUP66_CENG313` if the runtime changed,
and run its cells. For a private repository, authenticate using Colab's
GitHub integration or upload a repository ZIP and extract it instead.

## Structure

- `notebooks/`: analyses, visualisations, evaluation and explanations.
- `src/data.py`: load pairs and create the real/fake EDA view.
- `src/utils.py`: text measurement helpers.
- `src/models.py`: baseline pipeline, predictions and cross-validation.
- `outputs/submissions/submission.csv`: generated submission (not tracked).
- `data/` and `.venv/`: local data and environment (not tracked).

## Results

| Baseline | Holdout accuracy | Mean 5-fold accuracy | Fold std (percentage points) |
|---|---:|---:|---:|
| Majority position | 52.63% | 51.58% | 2.35 |
| Shorter text | 63.16% | 73.68% | 6.45 |
| TF-IDF + Logistic Regression | 94.74% | 91.58% | 4.71 |

## Kaggle submission history

| Date | Model | Public score | Private score |
|---|---|---:|---:|
| 2026-10-21 | TF-IDF + Logistic Regression | 0.86721 | 0.84641 |

## Progress log

- Completed EDA of lengths, corrupted examples, non-ASCII ratios,
  lexical diversity and word overlap.
- Implemented three baselines, pair-level validation, five-fold evaluation
  and a Kaggle submission.
- Moved reusable functions into `src/` and removed machine-specific paths.

## Student information

GitHub username: `tosuun`. İbrahim Emir Tosun 23118080074
