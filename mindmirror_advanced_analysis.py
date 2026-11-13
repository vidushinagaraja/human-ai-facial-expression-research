import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr, spearmanr


# ==================================================
# MINDMIRROR — ADVANCED STATISTICAL ANALYSIS
# ==================================================

print("\n🧠 MINDMIRROR — ADVANCED ANALYSIS")

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "mindmirror_full_results.csv"

results = pd.read_csv(DATA_FILE)

# Keep only complete observations.
results = results.dropna(
    subset=[
        "actor_id",
        "intended_expression",
        "normalized_human_entropy",
        "normalized_ai_entropy"
    ]
).copy()

HUMAN = "normalized_human_entropy"
AI = "normalized_ai_entropy"

print(f"\nVideos analyzed: {len(results)}")
print(f"Unique actors: {results['actor_id'].nunique()}")


# ==================================================
# 1. OVERALL CORRELATIONS
# ==================================================

pearson_r, pearson_p = pearsonr(
    results[HUMAN],
    results[AI]
)

spearman_r, spearman_p = spearmanr(
    results[HUMAN],
    results[AI]
)

print("\n📊 OVERALL RESULTS")

print(f"Pearson r: {pearson_r:.4f}")
print(f"Pearson p-value: {pearson_p:.4f}")

print(f"Spearman rho: {spearman_r:.4f}")
print(f"Spearman p-value: {spearman_p:.4f}")


# ==================================================
# 2. ACTOR-LEVEL CLUSTER BOOTSTRAP
# ==================================================

print("\n🧠 RUNNING ACTOR-AWARE BOOTSTRAP")

# Recordings from the same actor can be related.
# We therefore resample actors, not individual videos.

rng = np.random.default_rng(42)

actors = results["actor_id"].unique()

actor_groups = {
    actor: group[[HUMAN, AI]].to_numpy()
    for actor, group in results.groupby("actor_id")
}

N_BOOTSTRAPS = 2000

bootstrap_correlations = []

for iteration in range(N_BOOTSTRAPS):

    # Sample actors with replacement.
    sampled_actors = rng.choice(
        actors,
        size=len(actors),
        replace=True
    )

    # Keep all recordings associated
    # with each sampled actor.
    sampled_data = np.vstack([
        actor_groups[actor]
        for actor in sampled_actors
    ])

    human = sampled_data[:, 0]
    ai = sampled_data[:, 1]

    # Skip samples with no variation.
    if np.std(human) == 0 or np.std(ai) == 0:
        continue

    correlation = np.corrcoef(
        human,
        ai
    )[0, 1]

    bootstrap_correlations.append(correlation)

bootstrap_correlations = np.array(
    bootstrap_correlations
)

# Percentile bootstrap confidence interval.
ci_lower, ci_upper = np.percentile(
    bootstrap_correlations,
    [2.5, 97.5]
)

print("\n📊 ACTOR-AWARE BOOTSTRAP RESULTS")

print(f"Original Pearson r: {pearson_r:.4f}")

print(
    f"95% Bootstrap CI: "
    f"[{ci_lower:.4f}, {ci_upper:.4f}]"
)

print(
    f"Successful bootstrap samples: "
    f"{len(bootstrap_correlations)}"
)


# ==================================================
# 3. EXPRESSION-SPECIFIC ANALYSIS
# ==================================================

print("\n🧠 EXPRESSION-SPECIFIC RESULTS")

expression_results = []

for expression, group in results.groupby(
    "intended_expression"
):

    if len(group) < 3:
        continue

    if (
        group[HUMAN].nunique() < 2
        or group[AI].nunique() < 2
    ):
        continue

    r, p = pearsonr(
        group[HUMAN],
        group[AI]
    )

    rho, spearman_p_value = spearmanr(
        group[HUMAN],
        group[AI]
    )

    expression_results.append({
        "expression": expression,
        "videos": len(group),
        "actors": group["actor_id"].nunique(),
        "mean_human_uncertainty": group[HUMAN].mean(),
        "mean_ai_uncertainty": group[AI].mean(),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": spearman_p_value
    })

expression_df = pd.DataFrame(
    expression_results
)

print(
    expression_df.round(4).to_string(index=False)
)

expression_df.to_csv(
    BASE_DIR / "mindmirror_expression_statistics.csv",
    index=False
)


# ==================================================
# 4. SAVE OVERALL STATISTICS
# ==================================================

overall_statistics = pd.DataFrame([{
    "videos": len(results),
    "actors": len(actors),
    "pearson_r": pearson_r,
    "pearson_p_naive": pearson_p,
    "spearman_rho": spearman_r,
    "spearman_p_naive": spearman_p,
    "cluster_bootstrap_ci_lower": ci_lower,
    "cluster_bootstrap_ci_upper": ci_upper,
    "bootstrap_iterations": len(bootstrap_correlations)
}])

overall_statistics.to_csv(
    BASE_DIR / "mindmirror_advanced_statistics.csv",
    index=False
)


# ==================================================
# 5. BOOTSTRAP DISTRIBUTION GRAPH
# ==================================================

plt.figure(figsize=(9, 6))

plt.hist(
    bootstrap_correlations,
    bins=35,
    color="#6658F5",
    alpha=0.8,
    edgecolor="white"
)

plt.axvline(
    pearson_r,
    color="black",
    linestyle="--",
    label=f"Observed r = {pearson_r:.3f}"
)

plt.axvline(
    ci_lower,
    color="red",
    linestyle=":",
    label="95% bootstrap interval"
)

plt.axvline(
    ci_upper,
    color="red",
    linestyle=":"
)

plt.axvline(
    0,
    color="gray",
    linestyle="-",
    alpha=0.6
)

plt.xlabel("Pearson Correlation")
plt.ylabel("Bootstrap Frequency")

plt.title(
    "MindMirror: Actor-Cluster Bootstrap"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "mindmirror_bootstrap.png",
    dpi=300
)

plt.close()


# ==================================================
# 6. EXPRESSION CORRELATION GRAPH
# ==================================================

plt.figure(figsize=(10, 6))

colors = [
    "#6658F5" if value >= 0 else "#F28A90"
    for value in expression_df["pearson_r"]
]

plt.bar(
    expression_df["expression"],
    expression_df["pearson_r"],
    color=colors
)

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.ylabel("Pearson Correlation")

plt.xlabel("Actor's Intended Expression")

plt.title(
    "MindMirror: Human–AI Uncertainty Correlation "
    "by Expression"
)

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "mindmirror_expression_correlations.png",
    dpi=300
)

plt.close()


# ==================================================
# 7. COMPLETE
# ==================================================

print("\n✅ ADVANCED ANALYSIS COMPLETE!")

print("\nFiles created:")

print("• mindmirror_advanced_statistics.csv")
print("• mindmirror_expression_statistics.csv")
print("• mindmirror_bootstrap.png")
print("• mindmirror_expression_correlations.png")