import numpy as np
import pandas as pd

from pathlib import Path
from transformers import pipeline


# ============================================
# MINDMIRROR — CONFIDENT DISAGREEMENT ANALYSIS
# ============================================

print("\n🧠 MINDMIRROR — CONFIDENT DISAGREEMENT")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "mindmirror_full_results.csv"

FRAME_FOLDER = BASE_DIR / "crema_frames"

OUTPUT_FILE = (
    BASE_DIR / "mindmirror_confident_disagreement.csv"
)

HIGH_SCORE_THRESHOLD = 0.80

AI_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad"
]


# ============================================
# LOAD EXISTING RESULTS
# ============================================

results = pd.read_csv(RESULTS_FILE)

print(f"Videos available: {len(results)}")


# ============================================
# RESUME SAVED ANALYSIS
# ============================================

if OUTPUT_FILE.exists():

    analyzed = pd.read_csv(OUTPUT_FILE)

    completed = set(analyzed["video"])

    print(f"Previously analyzed: {len(completed)}")

else:

    analyzed = pd.DataFrame()

    completed = set()


# ============================================
# LOAD EXISTING AI MODEL
# ============================================

print("\nLoading AI model...")

classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    device="cpu"
)

print("✅ AI model ready!")


# ============================================
# ANALYZE EVERY RECORDING
# ============================================

for index, row in results.iterrows():

    filename = row["video"]

    if filename in completed:
        continue

    image_path = (
        FRAME_FOLDER / f"{Path(filename).stem}.jpg"
    )

    if not image_path.exists():

        print(f"Missing frame: {filename}")
        continue

    predictions = classifier(
        str(image_path),
        top_k=7
    )

    score_dict = {
        item["label"].lower(): item["score"]
        for item in predictions
    }

    # Restrict predictions to the six
    # categories available to human observers.
    common_scores = np.array(
        [
            score_dict[label]
            for label in AI_LABELS
        ],
        dtype=float
    )

    common_scores /= common_scores.sum()

    winning_index = int(
        np.argmax(common_scores)
    )

    ai_label = AI_LABELS[winning_index]

    ai_score = float(
        common_scores[winning_index]
    )

    human_label = str(
        row["human_prediction"]
    ).lower()

    # Ties cannot be treated as a unique majority.
    unique_majority = (
        human_label in AI_LABELS
        and not bool(row["human_majority_tied"])
    )

    agreement = (
        ai_label == human_label
        if unique_majority
        else None
    )

    high_score = (
        ai_score >= HIGH_SCORE_THRESHOLD
    )

    confident_disagreement = (
        unique_majority
        and high_score
        and ai_label != human_label
    )

    new_row = pd.DataFrame([{
        "video": filename,
        "actor_id": row["actor_id"],
        "intended_expression":
            row["intended_expression"],
        "human_prediction": human_label,
        "human_entropy":
            row["normalized_human_entropy"],
        "ai_prediction": ai_label,
        "ai_top_score": ai_score,
        "unique_human_majority": unique_majority,
        "human_ai_agreement": agreement,
        "high_ai_score": high_score,
        "confident_disagreement":
            confident_disagreement
    }])

    analyzed = pd.concat(
        [analyzed, new_row],
        ignore_index=True
    )

    # Save after each image.
    analyzed.to_csv(
        OUTPUT_FILE,
        index=False
    )

    completed.add(filename)

    if (index + 1) % 25 == 0:

        print(
            f"Analyzed {index + 1}/{len(results)}"
        )


# ============================================
# FINAL SUMMARY
# ============================================

valid = analyzed[
    analyzed["unique_human_majority"] == True
].copy()

valid["human_ai_agreement"] = (
    valid["human_ai_agreement"]
    .astype(bool)
)

valid["confident_disagreement"] = (
    valid["confident_disagreement"]
    .astype(bool)
)

total = len(valid)

agreements = int(
    valid["human_ai_agreement"].sum()
)

disagreements = total - agreements

confident_errors = int(
    valid["confident_disagreement"].sum()
)

print("\n✅ CONFIDENT DISAGREEMENT ANALYSIS COMPLETE")

print(f"\nTotal recordings: {len(analyzed)}")
print(f"Recordings with unique human majority: {total}")

if total > 0:

    print(
        f"Human–AI agreement: "
        f"{agreements / total * 100:.1f}%"
    )

    print(
        f"Human–AI disagreement: "
        f"{disagreements / total * 100:.1f}%"
    )

    print(
        f"High-scoring disagreements: "
        f"{confident_errors}"
    )

    print(
        f"High-scoring disagreement rate: "
        f"{confident_errors / total * 100:.1f}%"
    )

print("\nResults saved to:")
print(OUTPUT_FILE.name)