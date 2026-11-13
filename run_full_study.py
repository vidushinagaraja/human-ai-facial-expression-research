import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from transformers import pipeline


# ==================================================
# MINDMIRROR — COMPLETE RESEARCH PIPELINE
# ==================================================

print("\n🧠 MINDMIRROR — FULL RESEARCH STUDY")

BASE_DIR = Path(__file__).resolve().parent
CREMA_DIR = Path.home() / "Desktop" / "crema_d_source"

VIDEO_DIR = CREMA_DIR / "VideoFlash"
RATINGS_FILE = BASE_DIR / "finishedResponses.csv"

RESULTS_FILE = BASE_DIR / "mindmirror_full_results.csv"
SAMPLE_FILE = BASE_DIR / "mindmirror_sample.csv"

FRAME_DIR = BASE_DIR / "crema_frames"
FRAME_DIR.mkdir(exist_ok=True)

# Reproducible research settings.
RANDOM_SEED = 42
VIDEOS_PER_EXPRESSION = 100

EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

HUMAN_CODES = {
    "A": "angry",
    "D": "disgust",
    "F": "fear",
    "H": "happy",
    "N": "neutral",
    "S": "sad",
}

AI_LABELS = list(HUMAN_CODES.values())


# ==================================================
# 1. CHECK REQUIRED FILES
# ==================================================

if not VIDEO_DIR.exists():
    raise FileNotFoundError(
        f"CREMA-D folder not found: {VIDEO_DIR}"
    )

if not RATINGS_FILE.exists():
    raise FileNotFoundError(
        f"Human ratings not found: {RATINGS_FILE}"
    )


# ==================================================
# 2. LOAD HUMAN JUDGMENTS
# ==================================================

print("\nLoading human judgments...")

ratings = pd.read_csv(
    RATINGS_FILE,
    low_memory=False
)

# Keep visual-only responses.
ratings = ratings[
    pd.to_numeric(
        ratings["queryType"],
        errors="coerce"
    ) == 2
].copy()

ratings["clipName"] = (
    ratings["clipName"]
    .astype(str)
    .str.strip()
    .str.removesuffix(".flv")
)

ratings = ratings[
    ratings["respEmo"].isin(HUMAN_CODES)
].copy()

human_groups = {
    name: group
    for name, group in ratings.groupby("clipName")
}

print(f"Visual-only ratings loaded: {len(ratings)}")


# ==================================================
# 3. SELECT A REPRODUCIBLE SAMPLE
# ==================================================

if SAMPLE_FILE.exists():

    sample = pd.read_csv(SAMPLE_FILE)

    print("\nUsing previously saved sample.")

else:

    rng = random.Random(RANDOM_SEED)

    selected = []

    all_videos = sorted(VIDEO_DIR.glob("*.flv"))

    for emotion_code in EMOTIONS:

        candidates = []

        for video in all_videos:

            parts = video.stem.split("_")

            if len(parts) != 4:
                continue

            if parts[2] != emotion_code:
                continue

            if video.stem not in human_groups:
                continue

            candidates.append(video.name)

        count = min(
            VIDEOS_PER_EXPRESSION,
            len(candidates)
        )

        selected.extend(
            rng.sample(candidates, count)
        )

    sample = pd.DataFrame({
        "video": selected
    })

    sample.to_csv(
        SAMPLE_FILE,
        index=False
    )

    print("\nResearch sample selected and saved.")

print(f"Videos selected: {len(sample)}")


# ==================================================
# 4. DOWNLOAD SELECTED VIDEOS
# ==================================================

print("\nChecking selected video downloads...")

selected_names = sample["video"].tolist()

# Git LFS is installed in your personal folder.
git_lfs = Path.home() / ".local" / "bin" / "git-lfs"

if not git_lfs.exists():
    raise FileNotFoundError(
        "Git LFS was not found in ~/.local/bin."
    )

# Git LFS accepts comma-separated include patterns.
include_patterns = ",".join(
    f"VideoFlash/{name}"
    for name in selected_names
)

subprocess.run(
    [
        str(git_lfs),
        "pull",
        f"--include={include_patterns}",
        "--exclude=AudioMP3/*,AudioWAV/*",
    ],
    cwd=CREMA_DIR,
    check=True
)

print("\n✅ Selected video download completed!")


# ==================================================
# 5. LOAD PREVIOUS RESULTS IF AVAILABLE
# ==================================================

if RESULTS_FILE.exists():

    results = pd.read_csv(RESULTS_FILE)

    completed = set(results["video"].tolist())

    print(
        f"\nResuming analysis: {len(completed)} "
        "videos already processed."
    )

else:

    results = pd.DataFrame()

    completed = set()


# ==================================================
# 6. LOAD AI MODEL
# ==================================================

print("\nLoading facial-expression AI...")

classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    device="cpu"
)

print("✅ AI model ready!")


# ==================================================
# 7. ANALYZE SELECTED VIDEOS
# ==================================================

for index, filename in enumerate(selected_names, start=1):

    if filename in completed:
        continue

    print(
        f"\n[{index}/{len(selected_names)}] "
        f"Analyzing {filename}"
    )

    video_path = VIDEO_DIR / filename
    clip_id = video_path.stem

    try:

        # ------------------------------------------
        # EXTRACT MIDDLE VIDEO FRAME
        # ------------------------------------------

        video = cv2.VideoCapture(
            str(video_path)
        )

        if not video.isOpened():
            print("⚠️ Cannot open video.")
            video.release()
            continue

        total_frames = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            total_frames // 2
        )

        success, frame = video.read()

        video.release()

        if not success:
            print("⚠️ Frame extraction failed.")
            continue

        frame_path = FRAME_DIR / f"{clip_id}.jpg"

        if not cv2.imwrite(str(frame_path), frame):
            print("⚠️ Could not save frame.")
            continue

        # ------------------------------------------
        # AI PREDICTIONS
        # ------------------------------------------

        predictions = classifier(
            str(frame_path),
            top_k=7
        )

        ai_scores = {
            item["label"].lower(): item["score"]
            for item in predictions
        }

        # Original seven-category AI distribution.
        all_scores = np.array(
            list(ai_scores.values()),
            dtype=float
        )

        all_scores /= all_scores.sum()

        ai_entropy_7 = -np.sum(
            all_scores * np.log2(all_scores + 1e-12)
        )

        # Match the six expression categories
        # available to human observers.
        common_scores = np.array(
            [ai_scores[label] for label in AI_LABELS],
            dtype=float
        )

        common_scores /= common_scores.sum()

        ai_entropy_6 = -np.sum(
            common_scores
            * np.log2(common_scores + 1e-12)
        )

        ai_prediction = AI_LABELS[
            int(np.argmax(common_scores))
        ]

        # ------------------------------------------
        # HUMAN JUDGMENTS
        # ------------------------------------------

        group = human_groups.get(clip_id)

        if group is None or group.empty:
            print("⚠️ No human ratings.")
            continue

        counts = group["respEmo"].value_counts()

        probabilities = counts / counts.sum()

        human_entropy = -np.sum(
            probabilities
            * np.log2(probabilities)
        )

        max_votes = counts.max()

        leaders = counts[
            counts == max_votes
        ].index.tolist()

        tied = len(leaders) > 1

        if tied:
            human_prediction = "tied"
            agreement = np.nan
        else:
            human_prediction = HUMAN_CODES[leaders[0]]
            agreement = (
                human_prediction == ai_prediction
            )

        actor_id = clip_id.split("_")[0]

        # ------------------------------------------
        # STORE RESULTS
        # ------------------------------------------

        new_row = pd.DataFrame([{
            "video": filename,
            "actor_id": actor_id,
            "intended_expression": EMOTIONS[
                clip_id.split("_")[2]
            ],
            "human_ratings": len(group),
            "human_prediction": human_prediction,
            "human_entropy": human_entropy,
            "ai_prediction": ai_prediction,
            "ai_entropy_6": ai_entropy_6,
            "ai_entropy_7": ai_entropy_7,
            "human_majority_tied": tied,
            "human_ai_agreement": agreement,
            "normalized_human_entropy":
                human_entropy / np.log2(6),
            "normalized_ai_entropy":
                ai_entropy_6 / np.log2(6),
        }])

        results = pd.concat(
            [results, new_row],
            ignore_index=True
        )

        # Save after every video so progress
        # survives an interruption.
        results.to_csv(
            RESULTS_FILE,
            index=False
        )

        completed.add(filename)

        print(
            f"Human entropy: {human_entropy:.3f}"
        )

        print(
            f"AI entropy: {ai_entropy_6:.3f}"
        )

    except Exception as error:
        print(f"⚠️ Skipped video: {error}")
        continue


# ==================================================
# 8. STATISTICAL ANALYSIS
# ==================================================

print("\n🧠 COMPUTING STATISTICAL RESULTS")

data = results[
    [
        "normalized_human_entropy",
        "normalized_ai_entropy"
    ]
].dropna()

if len(data) < 3:
    raise ValueError(
        "Not enough valid videos for correlation analysis."
    )

human = data["normalized_human_entropy"]
ai = data["normalized_ai_entropy"]

pearson_r, pearson_p = pearsonr(human, ai)

spearman_r, spearman_p = spearmanr(human, ai)

statistics = pd.DataFrame([
    {
        "test": "Pearson",
        "correlation": pearson_r,
        "p_value": pearson_p,
        "n": len(data)
    },
    {
        "test": "Spearman",
        "correlation": spearman_r,
        "p_value": spearman_p,
        "n": len(data)
    }
])

statistics.to_csv(
    BASE_DIR / "mindmirror_statistics.csv",
    index=False
)

print("\nSTATISTICAL RESULTS")
print(statistics.to_string(index=False))


# ==================================================
# 9. CREATE UNCERTAINTY GRAPH
# ==================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    human,
    ai,
    alpha=0.5,
    color="#6658F5"
)

plt.xlabel("Human Uncertainty (Normalized Entropy)")
plt.ylabel("AI Uncertainty (Normalized Entropy)")

plt.title(
    "MindMirror: Human vs AI Facial-Expression Uncertainty"
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "mindmirror_correlation.png",
    dpi=300
)

plt.close()


# ==================================================
# 10. CREATE EXPRESSION COMPARISON GRAPH
# ==================================================

expression_summary = (
    results.groupby("intended_expression")
    [
        [
            "normalized_human_entropy",
            "normalized_ai_entropy"
        ]
    ]
    .mean()
)

expression_summary.plot(
    kind="bar",
    figsize=(10, 6),
    color=["#6658F5", "#F28A90"]
)

plt.title(
    "MindMirror: Uncertainty by Intended Expression"
)

plt.xlabel("Actor's Intended Expression")
plt.ylabel("Mean Normalized Entropy")

plt.xticks(rotation=30)
plt.ylim(0, 1)

plt.legend([
    "Human uncertainty",
    "AI uncertainty"
])

plt.tight_layout()

plt.savefig(
    BASE_DIR / "mindmirror_expression_analysis.png",
    dpi=300
)

plt.close()


# ==================================================
# 11. COMPLETE
# ==================================================

print("\n✅ MINDMIRROR FULL ANALYSIS COMPLETE!")

print(f"\nVideos analyzed: {len(results)}")

print("\nFiles created:")

print("• mindmirror_sample.csv")
print("• mindmirror_full_results.csv")
print("• mindmirror_statistics.csv")
print("• mindmirror_correlation.png")
print("• mindmirror_expression_analysis.png")

print("\n🧠 Research pipeline finished!")