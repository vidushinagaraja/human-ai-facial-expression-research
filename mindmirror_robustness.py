import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from transformers import pipeline


# ==================================================
# MINDMIRROR — FINAL ROBUSTNESS ANALYSIS
# ==================================================

print("\n🧠 MINDMIRROR — ROBUSTNESS ANALYSIS")

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = (
    Path.home()
    / "Desktop"
    / "crema_d_source"
    / "VideoFlash"
)

RESULTS_FILE = (
    BASE_DIR / "mindmirror_full_results.csv"
)

FRAME_RESULTS_FILE = (
    BASE_DIR / "mindmirror_frame_robustness.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "mindmirror_robustness_summary.csv"
)

# Six expression categories shared with humans.
AI_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad"
]


# ==================================================
# 1. LOAD EXISTING RESEARCH RESULTS
# ==================================================

results = pd.read_csv(RESULTS_FILE)

print(f"\nOriginal videos: {len(results)}")
print(f"Unique actors: {results['actor_id'].nunique()}")

# These columns must be present.
required_columns = [
    "video",
    "actor_id",
    "intended_expression",
    "normalized_human_entropy",
    "normalized_ai_entropy"
]

missing = [
    column
    for column in required_columns
    if column not in results.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ==================================================
# 2. LOAD SAVED PROGRESS
# ==================================================

if FRAME_RESULTS_FILE.exists():

    frame_results = pd.read_csv(
        FRAME_RESULTS_FILE
    )

    completed = set(
        frame_results["video"].tolist()
    )

    print(
        f"Resuming: {len(completed)} "
        "videos already analyzed."
    )

else:

    frame_results = pd.DataFrame()

    completed = set()


# ==================================================
# 3. LOAD AI MODEL
# ==================================================

print("\nLoading AI model...")

classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    device="cpu"
)

print("✅ AI model ready!")


# ==================================================
# 4. DEFINE ENTROPY FUNCTION
# ==================================================

def calculate_ai_entropy(frame):

    """
    Run AI on one video frame and calculate
    normalized predictive entropy over the
    six shared expression categories.
    """

    # OpenCV uses BGR; convert to RGB.
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # The pipeline accepts a PIL image.
    from PIL import Image

    image = Image.fromarray(rgb_frame)

    predictions = classifier(
        image,
        top_k=7
    )

    score_dict = {
        item["label"].lower(): item["score"]
        for item in predictions
    }

    scores = np.array(
        [
            score_dict[label]
            for label in AI_LABELS
        ],
        dtype=float
    )

    # Renormalize the six-category scores.
    scores = scores / scores.sum()

    entropy = -np.sum(
        scores * np.log2(scores + 1e-12)
    )

    return entropy / np.log2(6)


# ==================================================
# 5. PROCESS EARLY AND LATE FRAMES
# ==================================================

for index, row in results.iterrows():

    filename = row["video"]

    if filename in completed:
        continue

    print(
        f"\n[{index + 1}/{len(results)}] "
        f"{filename}"
    )

    video_path = VIDEO_DIR / filename

    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():
        print("⚠️ Could not open video.")
        video.release()
        continue

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < 3:
        print("⚠️ Video has too few frames.")
        video.release()
        continue

    frame_entropies = {}

    success_all = True

    # Analyze 25% and 75% positions.
    for label, fraction in [
        ("early", 0.25),
        ("late", 0.75)
    ]:

        frame_index = min(
            int(total_frames * fraction),
            total_frames - 1
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = video.read()

        if not success:
            success_all = False
            break

        try:

            frame_entropies[label] = (
                calculate_ai_entropy(frame)
            )

        except Exception as error:

            print(f"⚠️ AI error: {error}")
            success_all = False
            break

    video.release()

    if not success_all:
        print("⚠️ Skipping incomplete video.")
        continue

    # Reuse the previously calculated
    # middle-frame entropy.
    middle_entropy = float(
        row["normalized_ai_entropy"]
    )

    early_entropy = frame_entropies["early"]
    late_entropy = frame_entropies["late"]

    mean_entropy = np.mean([
        early_entropy,
        middle_entropy,
        late_entropy
    ])

    # Save the results.
    new_row = pd.DataFrame([{
        "video": filename,
        "actor_id": row["actor_id"],
        "intended_expression":
            row["intended_expression"],
        "human_entropy":
            row["normalized_human_entropy"],
        "ai_early": early_entropy,
        "ai_middle": middle_entropy,
        "ai_late": late_entropy,
        "ai_three_frame_mean": mean_entropy,
        "ai_frame_range": (
            max(
                early_entropy,
                middle_entropy,
                late_entropy
            )
            -
            min(
                early_entropy,
                middle_entropy,
                late_entropy
            )
        )
    }])

    frame_results = pd.concat(
        [frame_results, new_row],
        ignore_index=True
    )

    # Save after every video.
    frame_results.to_csv(
        FRAME_RESULTS_FILE,
        index=False
    )

    completed.add(filename)

    print(
        f"Early: {early_entropy:.3f} | "
        f"Middle: {middle_entropy:.3f} | "
        f"Late: {late_entropy:.3f}"
    )


# ==================================================
# 6. PREPARE STATISTICAL DATA
# ==================================================

data = frame_results.dropna(
    subset=[
        "human_entropy",
        "ai_middle",
        "ai_three_frame_mean",
        "intended_expression",
        "actor_id"
    ]
).copy()

print("\n✅ VIDEO PROCESSING COMPLETE")

print(f"Usable videos: {len(data)}")

if len(data) < 20:
    raise ValueError(
        "Too few usable recordings for robustness analysis."
    )


# ==================================================
# 7. CORRELATION FUNCTION
# ==================================================

def correlation_summary(x, y):

    pearson_r, pearson_p = pearsonr(
        x, y
    )

    spearman_r, spearman_p = spearmanr(
        x, y
    )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p
    }


# ==================================================
# 8. EXPRESSION-CATEGORY ADJUSTMENT
# ==================================================

def adjusted_correlation(dataset, ai_column):

    """
    Remove expression-category mean differences
    from human and AI entropy before correlating.
    """

    human_adjusted = (
        dataset["human_entropy"]
        -
        dataset.groupby(
            "intended_expression"
        )["human_entropy"].transform("mean")
    )

    ai_adjusted = (
        dataset[ai_column]
        -
        dataset.groupby(
            "intended_expression"
        )[ai_column].transform("mean")
    )

    return np.corrcoef(
        human_adjusted,
        ai_adjusted
    )[0, 1]


# ==================================================
# 9. ACTOR-CLUSTER BOOTSTRAP
# ==================================================

def cluster_bootstrap_adjusted(
    dataset,
    ai_column,
    iterations=2000
):

    rng = np.random.default_rng(42)

    actors = dataset["actor_id"].unique()

    actor_groups = {
        actor: group
        for actor, group
        in dataset.groupby("actor_id")
    }

    correlations = []

    for _ in range(iterations):

        sampled_actors = rng.choice(
            actors,
            size=len(actors),
            replace=True
        )

        sampled_data = pd.concat(
            [
                actor_groups[actor]
                for actor in sampled_actors
            ],
            ignore_index=True
        )

        correlation = adjusted_correlation(
            sampled_data,
            ai_column
        )

        if np.isfinite(correlation):
            correlations.append(correlation)

    return np.percentile(
        correlations,
        [2.5, 97.5]
    )


# ==================================================
# 10. COMPARE ORIGINAL AND THREE-FRAME RESULTS
# ==================================================

summary_rows = []

for name, column in [
    ("Middle frame", "ai_middle"),
    ("Three-frame mean", "ai_three_frame_mean")
]:

    stats = correlation_summary(
        data["human_entropy"],
        data[column]
    )

    adjusted_r = adjusted_correlation(
        data,
        column
    )

    ci_lower, ci_upper = (
        cluster_bootstrap_adjusted(
            data,
            column
        )
    )

    summary_rows.append({
        "method": name,
        "videos": len(data),
        "pearson_r": stats["pearson_r"],
        "spearman_rho": stats["spearman_rho"],
        "expression_adjusted_r": adjusted_r,
        "adjusted_cluster_ci_lower": ci_lower,
        "adjusted_cluster_ci_upper": ci_upper
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\n🧠 FINAL ROBUSTNESS RESULTS")

print(
    summary.round(4).to_string(index=False)
)


# ==================================================
# 11. FRAME STABILITY SUMMARY
# ==================================================

print("\n🤖 AI FRAME STABILITY")

print(
    "Mean within-video entropy range:",
    f"{data['ai_frame_range'].mean():.3f}"
)

print(
    "Median within-video entropy range:",
    f"{data['ai_frame_range'].median():.3f}"
)


# ==================================================
# 12. CREATE ROBUSTNESS GRAPH
# ==================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

# Original single-frame analysis.
axes[0].scatter(
    data["human_entropy"],
    data["ai_middle"],
    alpha=0.4,
    color="#6658F5"
)

axes[0].set_title(
    "Middle-Frame AI Uncertainty"
)

axes[0].set_xlabel(
    "Human Uncertainty"
)

axes[0].set_ylabel(
    "AI Uncertainty"
)

# Three-frame analysis.
axes[1].scatter(
    data["human_entropy"],
    data["ai_three_frame_mean"],
    alpha=0.4,
    color="#E8798B"
)

axes[1].set_title(
    "Three-Frame Mean AI Uncertainty"
)

axes[1].set_xlabel(
    "Human Uncertainty"
)

axes[1].set_ylabel(
    "AI Uncertainty"
)

for ax in axes:
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)

plt.suptitle(
    "MindMirror: Frame-Selection Robustness"
)

plt.tight_layout()

plt.savefig(
    BASE_DIR / "mindmirror_frame_robustness.png",
    dpi=300
)

plt.close()


# ==================================================
# 13. COMPLETE
# ==================================================

print("\n✅ ROBUSTNESS ANALYSIS COMPLETE!")

print("\nFiles created:")

print("• mindmirror_frame_robustness.csv")
print("• mindmirror_robustness_summary.csv")
print("• mindmirror_frame_robustness.png")