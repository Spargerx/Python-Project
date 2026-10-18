# Practice Progress and Learning Curve Analysis

A Python project for exploring how practice results change across sessions. It creates a reproducible sample dataset (or loads your own CSV), uses pandas and NumPy to summarize results, and uses ordinary Python conditions to recommend what to practice next.

## Run it

Install packages and run the script from this folder:

```bash
python -m pip install -r requirements.txt
python practice_progress.py
```

The script creates `data/practice_sessions.csv` and a `reports/` folder containing calculated session data, student/topic/overall summaries, and `practice_progress.png`.

### Options

```bash
python practice_progress.py --seed 5               # generate different sample data
python practice_progress.py --input sessions.csv   # analyze your own CSV instead
python practice_progress.py --output-dir results   # save data/ and reports/ elsewhere
```

A custom CSV must have these columns: `student`, `session_number`, `topic`, `questions_attempted`, `accuracy_pct`, `time_taken_minutes`, `hints_used`, `difficulty` (Easy, Medium or Hard). The script checks the file and stops with a clear message if columns are missing, values are out of range, or a student has duplicate session numbers.

## Dataset

The sample has 4 learners and 24 sessions each. Every row includes student, session number, topic, questions attempted, accuracy percentage, time taken in minutes, hints used, and difficulty. The generated learners have varied patterns so summaries can demonstrate steady improvement, inconsistent performance, and a possible plateau. Accuracy in the sample is also nudged slightly by topic and difficulty (for example, Hard sessions score a little lower), so the topic comparison is not pure noise.

## Analysis

- Sorts sessions by student and session number, then adds seconds per question, a difficulty-adjusted pace ratio, accuracy change, and a three-session rolling accuracy mean.
- Builds individual, topic, and overall summaries with `groupby` and calculated columns, reporting both simple mean accuracy and accuracy weighted by number of questions attempted.
- Estimates an accuracy improvement rate (percentage points per session) with a straight-line fit against the real session numbers.
- Classifies each learner as steady improvement, inconsistent performance, possible plateau, declining accuracy, or mixed/gradual progress, using simple rules on the last 6 sessions and the overall trend. Learners with fewer than 6 sessions are marked "Insufficient data".
- Recommends a next practice step with a plain Python function (`recommend`) using both accuracy (80% target) and speed (within 15% of the difficulty pace target). Learners already succeeding at Hard are told to maintain with mixed review, since there is no harder level.
- Gives each student an individual next-step recommendation based on their last 3 sessions.
- Saves a chart with three panels: rolling accuracy over time, average accuracy by topic, and average speed ratio per student.

All cutoffs (accuracy target, pace tolerance, plateau and consistency limits) are named constants at the top of `practice_progress.py`, so they are easy to adjust.

The accuracy and pace cutoffs are classroom examples, not universal standards. A plateau flag is only a description of recent results under the rule in this program, and difficulty changes between sessions can make recent results look noisier than they are. Nothing in this project measures intelligence.
