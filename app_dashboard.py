import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# HUMAN PERCEPTION AND ARTIFICIAL INTELLIGENCE
# RESEARCH DASHBOARD — REFINED DESIGN
# ============================================================

st.set_page_config(
    page_title="Human Perception and Artificial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DESIGN PALETTE
# ============================================================

NAVY = "#101C2B"
PANEL = "#19293B"
PANEL_LIGHT = "#21344A"

IVORY = "#F4F0E8"
MUTED = "#B8C6D4"

BLUE = "#86A9D2"
COPPER = "#E4AB8D"

GREEN = "#B7E4C7"
GREEN_BG = "#203D37"

RED = "#F0B6B6"
RED_BG = "#492D36"

GRID = "#34465A"


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Remove Streamlit header and toolbar */

    header[data-testid="stHeader"] {
        display: none !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    /* Main background */

    .stApp {
        background: #101C2B;
        color: #F4F0E8;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }

    /* Typography */

    h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.025em !important;
    }

    h1 {
        color: #F4F0E8 !important;
        font-size: 3rem !important;
        line-height: 1.2 !important;
        margin-bottom: 1.5rem !important;
    }

    h2 {
        color: #E4AB8D !important;
        font-size: 2.05rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        color: #86A9D2 !important;
        font-size: 1.4rem !important;
        margin-top: 1.3rem !important;
    }

    p, li {
        color: #D3DDE6;
        line-height: 1.85;
    }

    strong {
        color: #F4F0E8;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #162639;
        border-right: 1px solid #34465A;
    }

    section[data-testid="stSidebar"] * {
        color: #F4F0E8;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 3rem;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background: #19293B;
        border: 1px solid #34465A;
        border-radius: 10px;
        padding: 22px;
    }

    div[data-testid="stMetricLabel"] {
        color: #86A9D2;
    }

    div[data-testid="stMetricValue"] {
        color: #F4F0E8;
        font-family: Georgia, serif;
    }

    /* Dividers */

    hr {
        border-color: #34465A !important;
        margin-top: 2.1rem !important;
        margin-bottom: 2.1rem !important;
    }

    /* Dark Selectboxes */

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #21344A !important;
    border: 1px solid #526B85 !important;
    color: #F4F0E8 !important;
}

/* Selected option text */

div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #F4F0E8 !important;
}

/* Search text */

div[data-testid="stSelectbox"] input {
    color: #F4F0E8 !important;
    -webkit-text-fill-color: #F4F0E8 !important;
}

/* Dropdown options */

div[data-baseweb="popover"] {
    background-color: #21344A !important;
}

div[data-baseweb="popover"] li {
    background-color: #21344A !important;
    color: #F4F0E8 !important;
}

    div[data-baseweb="select"] span {
        color: #F4F0E8 !important;
    }

    /* Text inputs */

    input {
        color: #F4F0E8 !important;
    }

    /* Dataframes */

    div[data-testid="stDataFrame"] {
        border: 1px solid #465B70;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Hide widget link icons next to headings */

    a.header-anchor {
        display: none !important;
    }

    /* Small labels */

    .eyebrow {
        color: #86A9D2;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-size: 12px;
        margin-bottom: 18px;
    }

    /* Research callout */

    .research-box {
        background: #21344A;
        border-left: 4px solid #E4AB8D;
        padding: 30px 35px;
        border-radius: 8px;
        color: #F4F0E8;
        font-family: Georgia, serif;
        font-size: 23px;
        line-height: 1.65;
    }

    /* Important note */

    .research-note {
        background: #21344A;
        border: 1px solid #465B70;
        border-radius: 8px;
        padding: 20px 24px;
        color: #E2EAF1;
        line-height: 1.8;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Individual finding panels */

    .finding-card {
        background: #19293B;
        border-left: 3px solid #86A9D2;
        border-radius: 6px;
        padding: 22px 26px;
        margin-bottom: 17px;
    }

    .finding-number {
        color: #E4AB8D;
        font-family: Georgia, serif;
        font-size: 24px;
        margin-right: 12px;
    }

    .finding-title {
        color: #F4F0E8;
        font-size: 17px;
        font-weight: 600;
    }

    .finding-text {
        color: #C6D2DE;
        margin-top: 12px;
        line-height: 1.7;
    }

    /* Agreement labels */

    .status-agree {
        background: #203D37;
        color: #B7E4C7;
        padding: 5px 12px;
        border-radius: 5px;
    }

    .status-disagree {
        background: #492D36;
        color: #F0B6B6;
        padding: 5px 12px;
        border-radius: 5px;
    }
    /* Dropdown menu background */

[data-baseweb="popover"] {
    background-color: #21344A !important;
}

/* Dropdown list */

[data-baseweb="menu"],
[role="listbox"] {
    background-color: #21344A !important;
}

/* Dropdown options */

[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li,
[role="listbox"] [role="option"] {
    background-color: #21344A !important;
    color: #F4F0E8 !important;
}

/* Text inside dropdown options */

[data-baseweb="popover"] [role="option"] *,
[role="listbox"] [role="option"] * {
    color: #F4F0E8 !important;
}

/* Hover effect */

[data-baseweb="popover"] [role="option"]:hover,
[role="listbox"] [role="option"]:hover {
    background-color: #344C65 !important;
}
/* Fix white background inside expression selector */

div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: #21344A !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #21344A !important;
    color: #F4F0E8 !important;
}

/* Fix the internal search field */

div[data-testid="stSelectbox"] [data-baseweb="input"],
div[data-testid="stSelectbox"] [data-baseweb="input"] > div {
    background-color: transparent !important;
}

/* Keep typed and selected text visible */

div[data-testid="stSelectbox"] input {
    background-color: transparent !important;
    color: #F4F0E8 !important;
    -webkit-text-fill-color: #F4F0E8 !important;
    caret-color: #E4AB8D !important;
}

/* Remove default white focus background */

div[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #21344A !important;
    color: #F4F0E8 !important;
}

/* Fix placeholder text */

div[data-testid="stSelectbox"] input::placeholder {
    color: #B8C6D4 !important;
    -webkit-text-fill-color: #B8C6D4 !important;
}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    results = pd.read_csv(
        BASE_DIR / "mindmirror_full_results.csv"
    )

    confident = pd.read_csv(
        BASE_DIR / "mindmirror_confident_disagreement.csv"
    )

    expression_stats = pd.read_csv(
        BASE_DIR / "mindmirror_expression_statistics.csv"
    )

    robustness = pd.read_csv(
        BASE_DIR / "mindmirror_robustness_summary.csv"
    )

    return (
        results,
        confident,
        expression_stats,
        robustness
    )


try:

    (
        results,
        confident,
        expression_stats,
        robustness
    ) = load_data()

except FileNotFoundError as error:

    st.error(
        f"Research file not found: {error.filename}"
    )

    st.stop()


# ============================================================
# PREPARE RESULTS
# ============================================================

valid = confident[
    confident["unique_human_majority"] == True
].copy()

valid["human_ai_agreement"] = (
    valid["human_ai_agreement"] == True
)

total_recordings = len(results)
comparable_recordings = len(valid)
tied_recordings = total_recordings - comparable_recordings

agreement_rate = (
    valid["human_ai_agreement"].mean() * 100
)

disagreement_rate = 100 - agreement_rate

high_score_cases = valid[
    (valid["human_ai_agreement"] == False)
    &
    (valid["ai_top_score"] >= 0.80)
]

high_score_count = len(high_score_cases)

high_score_rate = (
    high_score_count / comparable_recordings * 100
)

pearson_r = results[
    "normalized_human_entropy"
].corr(
    results["normalized_ai_entropy"]
)


# ============================================================
# CHART STYLING
# ============================================================

def style_chart(ax):

    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(PANEL)

    ax.tick_params(
        colors=IVORY,
        labelsize=10
    )

    ax.xaxis.label.set_color(IVORY)
    ax.yaxis.label.set_color(IVORY)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.grid(
        color=IVORY,
        alpha=0.12
    )

    ax.set_axisbelow(True)


# ============================================================
# TABLE STYLING
# ============================================================

def style_results_table(df):

    """
    Keep table rows navy and highlight
    only the Agreement Status column.
    """

    def color_status(value):

        if value == "Agreement":

            return (
                "background-color: #DDF3E4; "
                "color: #245B43; "
                "font-weight: 600;"
            )

        elif value == "Disagreement":

            return (
                "background-color: #FBE1E2; "
                "color: #934650; "
                "font-weight: 600;"
            )

        return (
            "background-color: #21344A; "
            "color: #F4F0E8;"
        )

    styled = (
        df.style
        .set_properties(
            **{
                "background-color": "#19293B",
                "color": "#F4F0E8"
            }
        )
        .map(
            color_status,
            subset=["Agreement Status"]
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#21344A"),
                    ("color", "#F4F0E8")
                ]
            }
        ])
    )

    return styled


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <p class="eyebrow">
            Research Navigation
        </p>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Sections",
        [
            "Research Overview",
            "Human & Machine Judgments",
            "Confident Disagreement",
            "Uncertainty Analysis",
            "Methods & Limitations"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption(
        "Independent Computational Research"
    )

    st.caption(
        "Human Perception and Artificial Intelligence"
    )


# ============================================================
# PAGE 1 — RESEARCH OVERVIEW
# ============================================================

if page == "Research Overview":

    st.markdown(
        '<p class="eyebrow">'
        'Cognitive Science · Artificial Intelligence'
        '</p>',
        unsafe_allow_html=True
    )

    st.title(
        "When Human Perception and "
        "Artificial Intelligence Diverge"
    )

    st.markdown(
        """
        An exploratory computational investigation of
        agreement, perceptual ambiguity, and predictive
        uncertainty in facial-expression categorization.
        """
    )

    st.divider()

    st.header("Research Question")

    st.markdown(
        """
        <div class="research-box">
            To what extent do AI facial-expression predictions
            and confidence patterns align with human
            perceptual judgments, and when do they diverge?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("The Study at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Facial Displays",
        total_recordings
    )

    c2.metric(
        "Unique Actors",
        results["actor_id"].nunique()
    )

    c3.metric(
        "Human–AI Agreement",
        f"{agreement_rate:.1f}%"
    )

    c4.metric(
        "High-Scoring Disagreement",
        f"{high_score_rate:.1f}%"
    )

    st.markdown(
        f"""
        <div class="research-note">
            <strong>Understanding the sample:</strong>
            All {total_recordings} recordings were used
            for the uncertainty analysis.
            Of these, {comparable_recordings} had a unique
            human majority judgment and were included
            in the agreement analysis.
            The remaining {tied_recordings} recordings
            had tied leading judgments.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Research Aims & Hypotheses")

    st.markdown(
        """
        This study examines three related questions:
        """
    )

    st.markdown(
        """
        **H1 — Perceptual Agreement**

        - How frequently do AI predictions align with
          the human majority judgment?
        - Does agreement vary across facial-expression
          categories?

        **H2 — Uncertainty Correspondence**

        - Is greater human perceptual disagreement
          associated with greater AI predictive
          uncertainty?
        - Does this relationship remain after
          accounting for expression categories?

        **H3 — High-Scoring Disagreement**

        - How often does the AI strongly favor
          a different expression from the human
          majority judgment?
        - Which categories show these discrepancies?
        """
    )

    st.markdown(
        """
        <div class="research-note">
            <strong>Research transparency:</strong>
            These are exploratory aims. The broader
            research framing was refined after the
            initial analyses and was not preregistered.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Principal Findings")

    findings = [
        (
            "01",
            "Human–AI Agreement",
            f"Human and AI top-category judgments agreed "
            f"in {agreement_rate:.1f}% of recordings "
            "with a unique human majority."
        ),
        (
            "02",
            "High-Scoring Disagreement",
            f"{high_score_count} recordings showed "
            "disagreement despite AI top-category "
            "scores of at least 80%."
        ),
        (
            "03",
            "Uncertainty Correspondence",
            f"The original correlation was very weak "
            f"(Pearson r = {pearson_r:.3f})."
        ),
        (
            "04",
            "Robustness",
            "After adjusting for intended expression "
            "category, the association became smaller. "
            "Actor-cluster bootstrap confidence "
            "intervals included zero."
        )
    ]

    for number, title, description in findings:

        st.markdown(
            f"""
            <div class="finding-card">
                <span class="finding-number">
                    {number}
                </span>
                <span class="finding-title">
                    {title}
                </span>
                <div class="finding-text">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 2 — HUMAN & MACHINE JUDGMENTS
# ============================================================

elif page == "Human & Machine Judgments":

    st.title("Human & Machine Judgments")

    st.markdown(
        """
        This section explores where human observers
        and the AI model assign the same expression
        category—and where they differ.
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Agreement",
        f"{agreement_rate:.1f}%"
    )

    c2.metric(
        "Disagreement",
        f"{disagreement_rate:.1f}%"
    )

    c3.metric(
        "Comparable Recordings",
        comparable_recordings
    )

    st.markdown(
        f"""
        <div class="research-note">
            <strong>Why {comparable_recordings} rather
            than {total_recordings}?</strong><br>
            A majority-label comparison requires one
            leading human expression category.
            In {tied_recordings} recordings, multiple
            expressions received the same highest
            number of votes. Those recordings were
            excluded from agreement calculations,
            but retained in uncertainty analyses.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("Agreement Across Expression Categories")

    emotion_agreement = (
        valid.groupby("intended_expression")
        ["human_ai_agreement"]
        .mean()
        .mul(100)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        emotion_agreement.index,
        emotion_agreement.values,
        color=BLUE
    )

    ax.set_xlabel("Agreement (%)")
    ax.set_xlim(0, 100)

    style_chart(ax)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **How to interpret this graph**

        - Each bar represents one intended expression
          category.
        - Longer bars indicate a higher proportion
          of AI predictions matching the human
          majority judgment.
        - Differences between categories may reflect
          model behavior, expression characteristics,
          or limitations of using still frames
          rather than full videos.
        """
    )

    st.divider()

    st.header("Explore Individual Recordings")

    selected_expression = st.selectbox(
        "Filter by intended expression",
        ["All expressions"]
        + sorted(
            valid["intended_expression"]
            .unique()
            .tolist()
        )
    )

    if selected_expression == "All expressions":
        filtered = valid.copy()
    else:
        filtered = valid[
            valid["intended_expression"]
            == selected_expression
        ].copy()

    filtered["Agreement Status"] = np.where(
        filtered["human_ai_agreement"],
        "Agreement",
        "Disagreement"
    )

    display_table = filtered[
        [
            "video",
            "human_prediction",
            "ai_prediction",
            "ai_top_score",
            "Agreement Status"
        ]
    ].rename(
        columns={
            "video": "Recording",
            "human_prediction": "Human Majority",
            "ai_prediction": "AI Prediction",
            "ai_top_score": "AI Top Score"
        }
    )

    st.dataframe(
        style_results_table(display_table),
        use_container_width=True,
        hide_index=True,
        column_config={
            "AI Top Score": st.column_config.ProgressColumn(
                "AI Top Score",
                min_value=0,
                max_value=1,
                format="%.2f"
            )
        }
    )

    st.markdown(
        """
        <div class="research-note">
            <span style="color:#B7E4C7;">
                Agreement
            </span>
            indicates matching categories.
            <br>
            <span style="color:#F0B6B6;">
                Disagreement
            </span>
            indicates different categories.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 3 — CONFIDENT DISAGREEMENT
# ============================================================

elif page == "Confident Disagreement":

    st.title("Confident Disagreement")

    st.markdown(
        """
        This section examines instances where the AI
        produces a high top-category score while
        selecting a different expression from the
        human majority judgment.
        """
    )

    st.divider()

    threshold = st.slider(
        "Minimum AI Top-Category Score",
        min_value=50,
        max_value=99,
        value=80,
        step=1,
        format="%d%%"
    )

    disagreements = valid[
        (valid["human_ai_agreement"] == False)
        &
        (valid["ai_top_score"] >= threshold / 100)
    ].copy()

    rate = (
        len(disagreements)
        / comparable_recordings * 100
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High-Scoring Disagreements",
        len(disagreements)
    )

    c2.metric(
        "Share of Comparable Recordings",
        f"{rate:.1f}%"
    )

    c3.metric(
        "Selected Threshold",
        f"{threshold}%"
    )

    st.markdown(
        """
        <div class="research-note">
            <strong>Important distinction:</strong>
            An AI top-category score of 80% does not
            necessarily mean an 80% probability of
            being correct. These are model output
            scores, and their calibration has not
            been established.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("Disagreement by Expression")

    emotion_counts = (
        disagreements["intended_expression"]
        .value_counts()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        emotion_counts.index,
        emotion_counts.values,
        color=RED
    )

    ax.set_xlabel(
        "Number of High-Scoring Disagreements"
    )

    style_chart(ax)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **Interpretation**

        - The bars show how many recordings meet
          the selected disagreement threshold.
        - Higher counts identify categories worth
          investigating further.
        - Counts should be interpreted alongside
          the number of comparable recordings
          in each category.
        """
    )

    st.divider()

    st.header("Explore Disagreement Cases")

    disagreements = disagreements.sort_values(
        "ai_top_score",
        ascending=False
    )

    disagreements["Agreement Status"] = (
        "Disagreement"
    )

    display_table = disagreements[
        [
            "video",
            "human_prediction",
            "ai_prediction",
            "ai_top_score",
            "Agreement Status"
        ]
    ].rename(
        columns={
            "video": "Recording",
            "human_prediction": "Human Majority",
            "ai_prediction": "AI Prediction",
            "ai_top_score": "AI Top Score"
        }
    )

    st.dataframe(
        style_results_table(display_table),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Recording identifiers are shown without "
        "redistributing facial images."
    )


# ============================================================
# PAGE 4 — UNCERTAINTY ANALYSIS
# ============================================================

elif page == "Uncertainty Analysis":

    st.title("Uncertainty Analysis")

    st.markdown(
        """
        Human disagreement and AI predictive uncertainty
        represent related but distinct quantities.

        Human uncertainty measures how widely observers'
        expression judgments are distributed.

        AI uncertainty measures how widely the model
        distributes its classification scores.
        """
    )

    st.divider()

    st.header("01 / Human–AI Uncertainty Relationship")

    st.metric(
        "Original Pearson Correlation",
        f"{pearson_r:.3f}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        results["normalized_human_entropy"],
        results["normalized_ai_entropy"],
        alpha=0.40,
        color=BLUE,
        s=42
    )

    ax.set_xlabel(
        "Human Perceptual Uncertainty"
    )

    ax.set_ylabel(
        "AI Predictive Uncertainty"
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    style_chart(ax)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f"""
        **What does this graph show?**

        - Each dot represents one facial-expression recording.
        - Moving right means human observers disagreed more.
        - Moving upward means the AI distributed its
          prediction scores more broadly.
        - If both systems showed similar uncertainty
          patterns, we would expect a clearer upward trend.

        **Observed finding**

        The correlation was only **r = {pearson_r:.3f}**,
        indicating a very weak positive relationship.

        Human disagreement therefore explained little
        of the variation in AI uncertainty through
        this simple linear comparison.
        """
    )

    st.divider()

    st.header("02 / Uncertainty Across Expressions")

    means = (
        results.groupby("intended_expression")
        [
            [
                "normalized_human_entropy",
                "normalized_ai_entropy"
            ]
        ]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(means))
    width = 0.35

    ax.bar(
        x - width / 2,
        means["normalized_human_entropy"],
        width,
        color=BLUE,
        label="Human"
    )

    ax.bar(
        x + width / 2,
        means["normalized_ai_entropy"],
        width,
        color=COPPER,
        label="AI"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(means.index)

    ax.set_ylim(0, 1)

    ax.set_ylabel(
        "Mean Normalized Entropy"
    )

    style_chart(ax)

    legend = ax.legend(
        facecolor=PANEL,
        edgecolor=GRID
    )

    for text in legend.get_texts():
        text.set_color(IVORY)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **What does this comparison reveal?**

        - Human observers showed relatively high
          uncertainty for sadness and fear.
        - The AI showed its highest average uncertainty
          for disgust.
        - Both systems showed relatively low
          uncertainty for happiness.

        These differences suggest that human and
        machine categorization difficulty may vary
        across expression types.

        The categories represent intended expressions,
        not verified internal emotions.
        """
    )

    st.divider()

    st.header("03 / Robustness Analysis")

    display_robustness = robustness.rename(
        columns={
            "method": "Frame Method",
            "videos": "Recordings",
            "pearson_r": "Original r",
            "spearman_rho": "Spearman rho",
            "expression_adjusted_r": "Adjusted r",
            "adjusted_cluster_ci_lower": "CI Lower",
            "adjusted_cluster_ci_upper": "CI Upper"
        }
    )

    st.dataframe(
        display_robustness.round(4).style.set_properties(
            **{
                "background-color": PANEL,
                "color": IVORY
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        **Why was this analysis necessary?**

        An overall correlation might be influenced
        by systematic differences between expression
        categories.

        The analysis controlled for intended expression
        category and examined the stability of the
        association using middle-frame predictions
        and mean uncertainty across three sampled frames.

        **Key Findings**

        - Middle-frame adjusted correlation: **0.028**
        - Three-frame adjusted correlation: **0.020**
        - Both actor-cluster confidence intervals
          included zero.

        These findings provide limited evidence of
        a reliable within-category association between
        human disagreement and AI uncertainty.

        **Frame-selection sensitivity**

        AI uncertainty also varied across frames
        within the same recording.

        The mean normalized entropy range across
        the three sampled frames was **0.286**.

        This highlights the importance of considering
        temporal information when interpreting
        facial-expression predictions.
        """
    )


# ============================================================
# PAGE 5 — METHODS & LIMITATIONS
# ============================================================

elif page == "Methods & Limitations":

    st.title("Methods & Limitations")

    st.markdown(
        """
        A transparent account of the dataset,
        computational procedures, statistical
        analyses, and limitations.
        """
    )

    st.divider()

    st.header("01 / Why This Research Matters")

    st.markdown(
        """
        AI systems increasingly generate predictions
        from human facial expressions.

        However, the interpretation of facial expressions
        is not always straightforward—even among
        human observers.

        This raises important questions about how
        AI predictions should be interpreted.

        **This research matters because:**

        - Human observers can disagree about
          the expression displayed in the same recording.
        - An AI model may strongly favor a classification
          that differs from human consensus.
        - Model scores are not automatically indicators
          of human perceptual agreement.
        - Human-centered AI systems should communicate
          predictive uncertainty responsibly.
        - Understanding disagreement can help identify
          limitations in automated facial-expression
          recognition.

        The study contributes to broader discussions
        in computational cognitive science, affective
        computing, and human-computer interaction.
        """
    )

    st.divider()

    st.header("02 / Dataset")

    st.markdown(
        """
        **Dataset: CREMA-D**

        The study used:

        - 600 facial-expression recordings.
        - 91 unique actors.
        - Six intended expression categories.
        - 100 recordings per category.
        - Existing human visual-only judgments.

        A reproducible sampling procedure was used
        to select recordings from the available dataset.
        """
    )

    st.header("03 / Artificial Intelligence Model")

    st.markdown(
        """
        **Model:** `trpakov/vit-face-expression`

        - Pretrained Vision Transformer.
        - Applied to extracted video frames.
        - Originally produces seven expression scores.
        - Scores were restricted and renormalized to
          the six expression categories shared with
          the human judgment data.
        - No new model training was conducted.
        """
    )

    st.header("04 / Quantifying Uncertainty")

    st.markdown(
        """
        Shannon entropy was used to quantify
        the spread of both human judgments
        and AI prediction scores.

        - Lower entropy indicates a more concentrated
          response distribution.
        - Higher entropy indicates a more distributed
          response pattern.
        - Entropy was normalized to a 0–1 scale.

        This allows the two distributions to be
        compared descriptively, although they
        represent different measurement processes.
        """
    )

    st.latex(
        r"H(X)=-\sum_{i=1}^{k}p_i\log_2(p_i)"
    )

    st.header("05 / Statistical Procedures")

    st.markdown(
        """
        The analysis included:

        - Pearson correlations.
        - Spearman correlations.
        - Expression-specific comparisons.
        - Expression-category adjustment.
        - Actor-cluster bootstrapping.
        - Alternative frame-selection analysis.
        - Exploratory high-scoring disagreement analysis.

        Statistical interpretations were based on
        both effect sizes and uncertainty estimates.
        """
    )

    st.divider()

    st.header("06 / Limitations")

    st.markdown(
        """
        **Video versus still-image comparison**

        - Human observers rated full recordings.
        - The AI analyzed sampled still frames.
        - This difference may influence agreement
          and uncertainty measurements.

        **Acted facial expressions**

        - The recordings portray instructed expressions.
        - Findings may not generalize to spontaneous
          facial behavior.

        **Model scores**

        - The AI's output scores have not been
          established as calibrated probabilities.
        - High scores do not establish correctness.

        **Expression versus emotional state**

        - Facial expressions do not conclusively
          reveal someone's internal emotions.
        - This research examines perceived expression
          categories rather than actual emotional states.

        **Sampling and generalizability**

        - Results come from one dataset and one model.
        - Other populations, models, and settings
          may produce different findings.

        **Exploratory design**

        - The broader research question was refined
          after initial data analysis.
        - Findings should be treated as exploratory,
          not preregistered confirmatory results.
        """
    )

    st.divider()

    st.header("07 / References")

    st.markdown(
        """
        - Cao et al. (2014). *CREMA-D: Crowd-sourced
          Emotional Multimodal Actors Dataset.*
          IEEE Transactions on Affective Computing.

        - [CREMA-D Repository](https://github.com/CheyneyComputerScience/CREMA-D)

        - [Vision Transformer Model](https://huggingface.co/trpakov/vit-face-expression)
        """
    )