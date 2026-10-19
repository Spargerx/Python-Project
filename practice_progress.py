#practice progress and learning curve analysis...

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TOPICS = ["Fractions", "Algebra", "Geometry", "Word Problems"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_TARGET_SECONDS = {"Easy": 90, "Medium": 150, "Hard": 240}
TOPIC_ACCURACY_OFFSET = {"Fractions": 2, "Algebra": 0, "Geometry": -1.5, "Word Problems": -3}
DIFFICULTY_ACCURACY_OFFSET = {"Easy": 3, "Medium": 0, "Hard": -4}

ACCURACY_THRESHOLD = 80
PACE_TOLERANCE = 1.15
RECENT_WINDOW = 6
MIN_SESSIONS_FOR_TREND = 6
STEADY_SLOPE_PP = 0.6
DECLINE_SLOPE_PP = -0.3
PLATEAU_GAIN_PP = 3
PLATEAU_RANGE_PP = 12
INCONSISTENT_STD_PP = 10

REQUIRED_COLUMNS = {"student","session_number","topic","questions_attempted","accuracy_pct","time_taken_minutes","hints_used","difficulty"}



def build_practice_dataset(seed=17):
    rng = np.random.default_rng(seed)
    learners = {"Arnav":("steady", 48),
                "Mohit":("inconsistent", 53), 
                "Mehul":("plateau", 47), 
                "Kakashi":("improving", 42)
                }
    rows = []
    for learner, (pattern, starting_accuracy) in learners.items():
        for session in range(1, 25):
            topic = TOPICS[(session - 1)%len(TOPICS)]
            difficulty = DIFFICULTIES[(session+1)%len(DIFFICULTIES)]
            if pattern == "steady":
                expected, noise = starting_accuracy+session*1.15, rng.normal(0, 3)
            elif pattern == "inconsistent":
                expected, noise = starting_accuracy+session*0.55, rng.normal(0, 13)
            elif pattern == "plateau":
                expected, noise = starting_accuracy+min(session, 10)*1.4, rng.normal(0, 3)
            else:
                expected, noise = starting_accuracy+session * 1.8, rng.normal(0, 4)
            expected += TOPIC_ACCURACY_OFFSET[topic]+DIFFICULTY_ACCURACY_OFFSET[difficulty]
            accuracy = float(np.clip(expected + noise, 20, 98))
            ques = int(rng.integers(8, 21))
            hints = int(rng.poisson(max(0.15, (100-accuracy)/24)))
            target = DIFFICULTY_TARGET_SECONDS[difficulty]
            seconds_per_question = target*rng.uniform(0.72, 1.40)
            if pattern == "improving":
                seconds_per_question *= max(0.68, 1.08 - session * 0.012)
            elif pattern == "inconsistent":
                seconds_per_question *= rng.uniform(0.7, 1.45)
            rows.append({"student" : learner,
                        "session_number" : session, 
                        "topic" : topic,
                        "questions_attempted" : ques, 
                        "accuracy_pct" : round(accuracy, 1),
                        "time_taken_minutes" : round(ques*seconds_per_question/60, 2),
                        "hints_used" : hints, "difficulty": difficulty
                        })
    return pd.DataFrame(rows)



def validate_dataset(data):
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input data has no rows.")
    unknown = set(data["difficulty"].dropna())-set(DIFFICULTY_TARGET_SECONDS)
    if unknown:
        raise ValueError(f"Unknown difficulty levels {sorted(unknown)}; "
                         f"expected {list(DIFFICULTY_TARGET_SECONDS)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Input data contains missing values...")
    if not data["accuracy_pct"].between(0,100).all():
        raise ValueError("accuracy_pct must be between 0 and 100...")
    if (data["questions_attempted"] <= 0).any() or (data["time_taken_minutes"] <= 0).any():
        raise ValueError("questions_attempted and time_taken_minutes must be positive...")
    if data.duplicated(["student", "session_number"]).any():
        raise ValueError("Duplicate (student, session_number) rows found...")




def recommend(accuracy_pct, speed_ratio, difficulty):
    accurate = accuracy_pct >= ACCURACY_THRESHOLD
    on_pace = speed_ratio <= PACE_TOLERANCE
    if accurate and on_pace:
        if difficulty == DIFFICULTIES[-1]:
            return "Wohoo!!! You've Mastered the top level; maintain with mixed review..."
        return "Move up a difficulty"
    if not accurate and not on_pace:
        return "Take a deep breath!!! Review the concept; use untimed practice..."
    if not accurate:
        return "Review errors; try a few more practice questions"
    return "Practice for fluency at the same difficulty"



def add_calculated_columns(data):
    result = data.sort_values(["student", "session_number"]).reset_index(drop=True)
    result["seconds_per_question"] = result["time_taken_minutes"]*60 / result["questions_attempted"]
    result["target_seconds_per_question"] = result["difficulty"].map(DIFFICULTY_TARGET_SECONDS)
    result["speed_ratio"] = result["seconds_per_question"] / result["target_seconds_per_question"]
    result["correct_answers"] = result["accuracy_pct"]/100 * result["questions_attempted"]
    result["accuracy_change"] = result.groupby("student")["accuracy_pct"].diff()
    result["rolling_3_accuracy"] = result.groupby("student")["accuracy_pct"].transform(lambda values: values.rolling(window=3, min_periods=1).mean())

    result["on_pace"] = result["speed_ratio"] <= PACE_TOLERANCE
    result["recommendation"] = result.apply(lambda row: recommend(row["accuracy_pct"], row["speed_ratio"], row["difficulty"]), axis=1)
    return result


def classify_progress(student_sessions):
    ordered = student_sessions.sort_values("session_number")
    accuracy = ordered["accuracy_pct"].to_numpy(dtype=float)
    session_numbers = ordered["session_number"].to_numpy(dtype=float)

    if len(accuracy) < MIN_SESSIONS_FOR_TREND:
        return {"progress_pattern": "Insufficient data",
                "accuracy_improvement_pp_per_session": np.nan,
                "recent_accuracy_std_dev": np.nan,
                "recent_accuracy_gain_pp": np.nan}

    slope = float(np.polyfit(session_numbers, accuracy, 1)[0])
    recent = accuracy[-RECENT_WINDOW:]
    variability = float(np.std(recent, ddof=1))
    gain = float(np.mean(recent[-3:])-np.mean(recent[:3]))
    plateau = abs(gain) < PLATEAU_GAIN_PP and np.ptp(recent) < PLATEAU_RANGE_PP
    inconsistent = variability >= INCONSISTENT_STD_PP or (np.ptp(recent) >= 25 and variability >= 8)

    if inconsistent:
        pattern = "Inconsistent performance"
    elif plateau:
        pattern = "Possible plateau"
    elif slope >= STEADY_SLOPE_PP:
        pattern = "Steady improvement"
    elif slope <= DECLINE_SLOPE_PP:
        pattern = "Declining accuracy"
    else:
        pattern = "Mixed or gradual progress"
    return {"progress_pattern": pattern,
            "accuracy_improvement_pp_per_session": round(slope, 2),
            "recent_accuracy_std_dev": round(variability, 2),
            "recent_accuracy_gain_pp": round(gain, 2)
            }


def make_summaries(data):
    student = data.groupby("student").agg(
        sessions=("session_number", "count"),
        questions=("questions_attempted", "sum"),
        correct=("correct_answers", "sum"),
        mean_accuracy_pct=("accuracy_pct", "mean"),
        mean_minutes=("time_taken_minutes", "mean"),
        hints_per_session=("hints_used", "mean"),
        mean_seconds_per_question=("seconds_per_question", "mean"),
        mean_speed_ratio=("speed_ratio", "mean"),
        on_pace_share=("on_pace", "mean")
    )
    student["weighted_accuracy_pct"] = student["correct"] / student["questions"] * 100
    student["on_pace_share"] *= 100
    student = student.drop(columns="correct").rename(columns={"on_pace_share": "on_pace_pct"})
    progress = pd.DataFrame.from_dict({name: classify_progress(group) for name, group in data.groupby("student")}, orient="index")
    latest = {}
    for name, group in data.groupby("student"):
        last3 = group.sort_values("session_number").tail(3)
        latest[name] = recommend(last3["accuracy_pct"].mean(),
                                 last3["speed_ratio"].mean(),
                                 last3["difficulty"].iloc[-1]
                                )
    student["next_step_recommendation"] = pd.Series(latest)
    student = student.join(progress).round(2).reset_index()

    topic = data.groupby("topic").agg(
        sessions=("session_number", "count"),
        questions=("questions_attempted", "sum"),
        correct=("correct_answers", "sum"),
        mean_accuracy_pct=("accuracy_pct", "mean"),
        mean_minutes=("time_taken_minutes", "mean"),
        mean_hints=("hints_used", "mean"),
        mean_seconds_per_question=("seconds_per_question", "mean")
    )
    topic["weighted_accuracy_pct"] = topic["correct"] / topic["questions"] * 100
    topic = topic.drop(columns="correct").round(2).reset_index()
    topic = topic.sort_values("mean_accuracy_pct", ascending=False)

    overall = pd.DataFrame([{
        "students": data["student"].nunique(),
        "sessions": len(data),
        "questions": int(data["questions_attempted"].sum()),
        "mean_accuracy_pct": round(data["accuracy_pct"].mean(), 2),
        "weighted_accuracy_pct": round(data["correct_answers"].sum()/ data["questions_attempted"].sum() * 100, 2),
        "mean_minutes_per_session": round(data["time_taken_minutes"].mean(), 2),
        "mean_hints_per_session": round(data["hints_used"].mean(), 2)
    }])
    return student, topic, overall



def save_charts(data, topic_summary, student_summary, report_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for student, group in data.groupby("student"):
        ordered = group.sort_values("session_number")
        axes[0].plot(ordered["session_number"], ordered["rolling_3_accuracy"],marker=".", linewidth=1.8, label=student)
    axes[0].set(title="Accuracy progress",xlabel="Session",ylabel="Accuracy %", ylim=(0, 100))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ordered_topics = topic_summary.sort_values("mean_accuracy_pct")
    axes[1].barh(ordered_topics["topic"], ordered_topics["mean_accuracy_pct"], color="#4c78a8")
    axes[1].set(title="Average accuracy by topic", xlabel="Accuracy %", xlim=(0, 100))
    axes[1].grid(axis="x", alpha=0.25)

    pace = student_summary.sort_values("mean_speed_ratio")
    axes[2].barh(pace["student"], pace["mean_speed_ratio"], color="#f58518")
    axes[2].axvline(PACE_TOLERANCE, color="red", linestyle="--", linewidth=1,label=f"On-pace limit ({PACE_TOLERANCE})")
    axes[2].set(title="Average speed ratio", xlabel="Speed ratio")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(report_dir / "practice_progress.png", dpi=160)
    plt.close(fig)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Practice progress and learning curve analysis")
    parser.add_argument("--seed", type=int, default=17, help="seed for the sample data")
    parser.add_argument("--input", type=Path, default=None, help="CSV of real sessions (skips sample-data generation)")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="folder that will hold data/ and reports/")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.output_dir / "data"
    report_dir = args.output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        raw = pd.read_csv(args.input)
    else:
        raw = build_practice_dataset(args.seed)
        raw.to_csv(data_dir / "practice_sessions.csv", index=False)
    validate_dataset(raw)

    analyzed = add_calculated_columns(raw)
    students, topics, overall = make_summaries(analyzed)
    analyzed.to_csv(report_dir / "analyzed_sessions.csv", index=False)
    students.to_csv(report_dir / "student_summary.csv", index=False)
    topics.to_csv(report_dir / "topic_summary.csv", index=False)
    overall.to_csv(report_dir / "overall_summary.csv", index=False)
    save_charts(analyzed, topics, students, report_dir)

    print("Overall summary\n", overall.to_string(index=False))
    print("\nStudent progress\n", students[["student", "mean_accuracy_pct", "weighted_accuracy_pct", "progress_pattern", "accuracy_improvement_pp_per_session","next_step_recommendation"]].to_string(index=False))
    print("\nTopic comparison\n", topics[["topic", "mean_accuracy_pct","mean_seconds_per_question"]].to_string(index=False))
    print(f"\nFiles saved in {data_dir} and {report_dir}.")
    print("Progress patterns describe practice results only; they do not measure intelligence.")


if __name__ == "__main__":
    main()
