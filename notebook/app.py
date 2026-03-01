"""
╔══════════════════════════════════════════════════════════════════════════╗
║   OULAD Student Burnout & Dropout Risk Dashboard                        ║
║   Run:  streamlit run app.py                                             ║
║   Requires:  data.csv  GB.pkl  (place in same directory as app.py)      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OULAD · Burnout Risk Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  – dark academic theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ───────────────────────────────────────── */
:root {
    --bg-base:        #0D1117;
    --bg-surface:     #161B22;
    --bg-elevated:    #1C2333;
    --bg-card:        #1E2A3A;
    --border:         #2D3748;
    --border-subtle:  #1F2937;
    --accent-blue:    #3B82F6;
    --accent-cyan:    #06B6D4;
    --accent-amber:   #F59E0B;
    --accent-red:     #EF4444;
    --accent-green:   #10B981;
    --accent-purple:  #8B5CF6;
    --text-primary:   #F0F4F8;
    --text-secondary: #94A3B8;
    --text-muted:     #64748B;
    --risk-low:       #10B981;
    --risk-medium:    #F59E0B;
    --risk-high:      #EF4444;
    --risk-critical:  #DC2626;
}

/* ── Global Reset ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    letter-spacing: -0.02em;
}

/* ── Metrics / KPI cards ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.15) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* ── Selectbox, sliders ──────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSlider [data-testid="stSlider"] > div { color: var(--accent-blue) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.1s;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Tabs ────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: var(--bg-surface) !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Custom Cards (via markdown) ─────────────────────────── */
.risk-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0;
    transition: transform 0.2s;
}
.risk-card:hover { transform: translateY(-2px); }

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-low    { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
.badge-high   { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-critical { background: rgba(220,38,38,0.20); color: #DC2626; border: 1px solid rgba(220,38,38,0.4); }

.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 2px;
}
.stat-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #F0F4F8;
    line-height: 1.1;
}
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #06B6D4; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #F0F4F8;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2D3748;
}

.intervention-card {
    background: var(--bg-elevated);
    border-left: 4px solid var(--accent-blue);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.intervention-card.urgent { border-left-color: var(--risk-high); }
.intervention-card.warning { border-left-color: var(--risk-medium); }
.intervention-card.ok { border-left-color: var(--risk-low); }

.sidebar-logo {
    text-align: center;
    padding: 0 1rem 1.5rem;
    border-bottom: 1px solid #2D3748;
    margin-bottom: 1.5rem;
}
.sidebar-logo .title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #F0F4F8;
    margin-top: 0.5rem;
}
.sidebar-logo .subtitle {
    font-size: 0.72rem;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.top-banner {
    background: linear-gradient(135deg, #1C2333 0%, #1E2A3A 50%, #1a2035 100%);
    border: 1px solid #2D3748;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.top-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS & MAPPINGS
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "activity_type_dataplus","activity_type_dualpane","activity_type_externalquiz",
    "activity_type_folder","activity_type_forumng","activity_type_glossary",
    "activity_type_homepage","activity_type_quiz","activity_type_resource",
    "gender_F","gender_M",
    "region_East Anglian Region","region_East Midlands Region","region_Ireland",
    "region_London Region","region_North Region","region_North Western Region",
    "region_Scotland","region_South East Region","region_South Region",
    "region_South West Region","region_Wales","region_West Midlands Region",
    "region_Yorkshire Region",
    "highest_education_A Level or Equivalent","highest_education_HE Qualification",
    "highest_education_Lower Than A Level",
    "imd_band_0-10%","imd_band_10-20","imd_band_20-30%","imd_band_30-40%",
    "imd_band_40-50%","imd_band_50-60%","imd_band_60-70%","imd_band_70-80%",
    "imd_band_80-90%","imd_band_90-100%",
    "age_band_0-35","age_band_35+",
    "disability_N","disability_Y",
    "final_result_Distinction","final_result_Fail","final_result_Pass","final_result_Withdrawn",
    "study_status_finished","study_status_unfinished",
    "withdrawal_status_didn't withdraw","withdrawal_status_early withdrawal",
    "withdrawal_status_late withdrawal","withdrawal_status_normal withdrawal",
    "sum","count","score","num_of_prev_attempts",
    "assessment_engagement_score","submission_timeliness","score_per_weight",
    "module_engagement_rate","repeat_student","performance_by_registration",
    "weighted_engagement","cumulative_score","engagement_consistency",
    "learning_pace","engagement_dropoff","activity_diversity","improvement_rate",
    "kmeans_cluster","average_engagement","engagement_classification",
]

# engagement_classification: 0=Moderate, 1=High, 2=Low  (LabelEncoder order)
ENGAGEMENT_LABEL = {0: "Moderate Risk", 1: "Low Risk", 2: "High Risk"}
RISK_COLOR = {
    "Low Risk":      "#10B981",
    "Moderate Risk": "#F59E0B",
    "High Risk":     "#EF4444",
}
RISK_BADGE = {
    "Low Risk":      "badge-low",
    "Moderate Risk": "badge-medium",
    "High Risk":     "badge-high",
}

KEY_BEHAVIORAL = [
    "sum", "count", "score", "assessment_engagement_score",
    "submission_timeliness", "module_engagement_rate",
    "activity_diversity", "improvement_rate", "engagement_consistency",
]
BEHAVIORAL_LABELS = {
    "sum":                       "Total VLE Clicks",
    "count":                     "Interaction Sessions",
    "score":                     "Avg Assessment Score",
    "assessment_engagement_score":"Engagement Score",
    "submission_timeliness":     "Submission Timeliness",
    "module_engagement_rate":    "Module Engagement Rate",
    "activity_diversity":        "Activity Diversity",
    "improvement_rate":          "Score Improvement Rate",
    "engagement_consistency":    "Engagement Consistency",
}

INTERVENTIONS = {
    "Low Risk": {
        "level_label": "✅ On Track",
        "color": "#10B981",
        "badge": "ok",
        "summary": "This student shows healthy engagement patterns. Maintain momentum with optional enrichment.",
        "actions": [
            "📊  Share a personalised progress report highlighting strong performance areas.",
            "🏆  Nominate for peer mentoring or study-group leadership roles.",
            "📚  Suggest optional advanced modules or challenge assessments.",
            "🎯  Set stretch learning goals for the next module presentation.",
        ]
    },
    "Moderate Risk": {
        "level_label": "⚠️ Monitor Closely",
        "color": "#F59E0B",
        "badge": "warning",
        "summary": "Declining engagement indicators detected. Timely nudges can prevent escalation.",
        "actions": [
            "📬  Send automated email check-in: 'We noticed your activity has changed — is everything OK?'",
            "🔗  Provide targeted VLE resource links matched to upcoming assessment topics.",
            "👥  Suggest joining a peer study group or online tutorial session.",
            "⏰  Enable automated submission deadline reminders (3 days, 1 day, day-of).",
            "📋  Flag to personal tutor for a brief supportive outreach call.",
        ]
    },
    "High Risk": {
        "level_label": "🚨 Immediate Action",
        "color": "#EF4444",
        "badge": "urgent",
        "summary": "Critical disengagement signals detected. Immediate human intervention is strongly recommended.",
        "actions": [
            "📞  URGENT: Escalate to academic counsellor for a scheduled welfare meeting within 48 hours.",
            "🧠  Conduct a structured check-in to identify external stressors (work, health, family).",
            "📅  Establish a weekly progress check-in schedule with a named support contact.",
            "🔄  Review if alternative assessment submission formats (extensions, deferrals) are applicable.",
            "💡  Explore whether a module deferral or reduced credit load is appropriate.",
            "📱  Enable 'Early Alert' flag in student information system for coordinated support.",
        ]
    }
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#94A3B8"),
    xaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(t=30, b=10, l=10, r=10),
)

# ─────────────────────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data.csv")
    df = pd.read_csv(data_path, index_col=0)
    return df

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "GB.pkl")
    return joblib.load(model_path)

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_risk_label(pred_class):
    return ENGAGEMENT_LABEL.get(int(pred_class), "Unknown")

def calculate_risk_score(probabilities):
    """Convert model probabilities → 0-100 risk score.
       Class mapping:  0=Moderate, 1=High, 2=Low
       High risk class (1) drives the score up; Low risk class (2) drives it down.
    """
    p_high     = probabilities[0][1]   # probability of High Engagement (Low burnout)
    p_moderate = probabilities[0][0]   # probability of Moderate Engagement
    p_low      = probabilities[0][2]   # probability of Low Engagement (High burnout)
    # Higher p_low  → higher risk score
    # Higher p_high → lower risk score
    raw = (p_low * 100) + (p_moderate * 50) + (p_high * 10)
    return round(min(raw, 100), 1)

def prepare_features(row, all_cols):
    """Ensure the row has exactly the columns the model was trained on."""
    row = row.copy()
    for c in all_cols:
        if c not in row.index:
            row[c] = 0
    return row[all_cols]

def get_deviation_data(student_row, global_means):
    deviations = {}
    for col in KEY_BEHAVIORAL:
        if col in student_row and col in global_means:
            s_val  = student_row[col]
            g_mean = global_means[col]
            std    = global_means.get(f"_std_{col}", 1)
            if std == 0:
                std = 1
            deviations[col] = {
                "student":   float(s_val),
                "global":    float(g_mean),
                "z_score":   float((s_val - g_mean) / std),
                "label":     BEHAVIORAL_LABELS.get(col, col),
            }
    return deviations

def build_gauge(risk_score, risk_label):
    color = RISK_COLOR.get(risk_label, "#94A3B8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Burnout Risk Score", "font": {"size": 14, "color": "#94A3B8", "family": "DM Sans"}},
        number={"suffix": "/100", "font": {"size": 36, "color": color, "family": "DM Serif Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2D3748",
                     "tickfont": {"color": "#64748B", "size": 10}},
            "bar":  {"color": color, "thickness": 0.18},
            "bgcolor": "#1C2333",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "rgba(16,185,129,0.12)"},
                {"range": [35, 65], "color": "rgba(245,158,11,0.12)"},
                {"range": [65,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": risk_score,
            },
        }
    ))
    fig.update_layout(**PLOTLY_THEME, height=260)
    return fig

def build_deviation_chart(deviations):
    labels    = [v["label"]   for v in deviations.values()]
    z_scores  = [v["z_score"] for v in deviations.values()]
    colors    = ["#EF4444" if z < -1 else "#F59E0B" if z < 0 else "#10B981" for z in z_scores]

    fig = go.Figure(go.Bar(
        x=z_scores, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{z:+.2f}σ" for z in z_scores],
        textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Z-score: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#2D3748", line_width=1.5)
    fig.update_layout(
        **PLOTLY_THEME,
        height=320,
        title=dict(text="Student vs. Cohort Average (Z-score deviation)", font=dict(size=13, color="#94A3B8")),
        xaxis_title="Standard Deviations from Mean",
        xaxis_range=[-3.5, 3.5],
        xaxis_gridcolor="#1F2937",
        xaxis_zerolinecolor="#374151"
    )
    return fig

def build_radar_chart(student_vals, mean_vals):
    cats = [BEHAVIORAL_LABELS[c] for c in KEY_BEHAVIORAL if c in student_vals]
    s    = [float(student_vals.get(c, 0)) for c in KEY_BEHAVIORAL if c in student_vals]
    m    = [float(mean_vals.get(c, 0))    for c in KEY_BEHAVIORAL if c in mean_vals]

    # normalise 0-1
    max_v = [max(abs(sv), abs(mv), 1e-9) for sv, mv in zip(s, m)]
    s_n   = [v / mx for v, mx in zip(s, max_v)]
    m_n   = [v / mx for v, mx in zip(m, max_v)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=m_n + [m_n[0]], theta=cats + [cats[0]],
        fill="toself", name="Cohort Avg",
        line_color="#3B82F6", fillcolor="rgba(59,130,246,0.08)",
        line_width=1.5,
    ))
    fig.add_trace(go.Scatterpolar(
        r=s_n + [s_n[0]], theta=cats + [cats[0]],
        fill="toself", name="This Student",
        line_color="#F59E0B", fillcolor="rgba(245,158,11,0.12)",
        line_width=2,
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1F2937",
                            tickfont=dict(size=9, color="#64748B")),
            angularaxis=dict(gridcolor="#1F2937", tickfont=dict(size=10, color="#94A3B8")),
        ),
        showlegend=True,
        height=320,
        title=dict(text="Engagement Profile Radar", font=dict(size=13, color="#94A3B8")),
    )
    return fig

def build_score_history_chart(df, student_id):
    """Simulate a score trend using cumulative_score if available."""
    row = df[df["id_student"] == student_id]
    if row.empty:
        return None
    score = float(row["score"].values[0])
    impr  = float(row["improvement_rate"].values[0])
    # generate 8-point synthetic history from final score + improvement rate
    points = [score + impr * (i - 7) for i in range(8)]
    points = [max(0, min(100, p)) for p in points]
    weeks  = [f"Wk {i+1}" for i in range(8)]
    cohort_avg = float(df["score"].mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=[cohort_avg] * 8,
        mode="lines", name="Cohort Avg",
        line=dict(color="#3B82F6", width=1.5, dash="dot"),
        hovertemplate="Cohort Avg: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=points,
        mode="lines+markers", name="This Student",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=7, color="#F59E0B",
                    line=dict(color="#1C2333", width=2)),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.05)",
        hovertemplate="Score: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        height=250,
        title=dict(text="Estimated Score Trajectory", font=dict(size=13, color="#94A3B8")),
        yaxis_range=[0, 105],
        yaxis_gridcolor="#1F2937"
    )
    return fig

def build_cohort_distribution(df, student_val, col):
    label = BEHAVIORAL_LABELS.get(col, col)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col], name="All Students",
        marker_color="#3B82F6", opacity=0.45,
        nbinsx=40,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=student_val, line_color="#F59E0B", line_width=2.5,
        annotation_text="  This Student", annotation_position="top right",
        annotation_font=dict(color="#F59E0B", size=11),
    )
    fig.update_layout(
        **PLOTLY_THEME,
        height=220,
        title=dict(text=f"Cohort Distribution: {label}", font=dict(size=12, color="#94A3B8")),
        bargap=0.05,
        showlegend=False,
    )
    return fig

def build_risk_donut(df):
    counts = df["engagement_classification"].value_counts()
    labels = [ENGAGEMENT_LABEL.get(k, str(k)) for k in counts.index]
    colors = [RISK_COLOR.get(l, "#94A3B8") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values,
        hole=0.62,
        marker=dict(colors=colors,
                    line=dict(color="#161B22", width=3)),
        textfont=dict(size=11, family="DM Sans"),
        hovertemplate="%{label}<br>%{value} students (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        height=260,
        title=dict(text="Cohort Risk Distribution", font=dict(size=13, color="#94A3B8")),
        showlegend=True,
        legend_orientation="h",
        legend_y=-0.08
    )
    return fig

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2rem;">🎓</div>
        <div class="title">OULAD Analytics</div>
        <div class="subtitle">Burnout & Dropout Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Navigation",
        ["📊  Dashboard — Existing Students", "🧪  Simulate — New Student"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.72rem; color:#64748B; letter-spacing:0.06em; text-transform:uppercase; font-weight:600; margin-bottom:0.5rem;">
        Model Info
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.7;">
        <span class="mono">GB.pkl</span> — Gradient Boosting<br>
        Test Accuracy: <b style="color:#10B981;">93.4%</b><br>
        Target: <span class="mono">engagement_classification</span><br>
        Classes: Low · Moderate · High Risk
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.72rem; color:#64748B;">
        Risk Thresholds
    </div>
    """, unsafe_allow_html=True)
    t1 = st.slider("Low / Medium boundary", 20, 50, 35, key="t1")
    t2 = st.slider("Medium / High boundary", 51, 80, 65, key="t2")

# ─────────────────────────────────────────────────────────────
#  LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
try:
    df   = load_data()
    model = load_model()
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    missing = str(e)

# ─────────────────────────────────────────────────────────────
#  TOP BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.9rem; color:#F0F4F8; line-height:1.1;">
                Student Burnout & Dropout Risk Monitor
            </div>
            <div style="font-size:0.85rem; color:#64748B; margin-top:0.3rem;">
                Open University Learning Analytics Dataset · Powered by Gradient Boosting
            </div>
        </div>
        <div style="display:flex; gap:2rem; flex-wrap:wrap;">
            <div><div class="stat-label">Dataset</div><div class="stat-value" style="font-size:1.2rem;">OULAD</div></div>
            <div><div class="stat-label">Students</div><div class="stat-value">23,343</div></div>
            <div><div class="stat-label">Courses</div><div class="stat-value">7</div></div>
            <div><div class="stat-label">Model Acc.</div><div class="stat-value" style="color:#10B981;">93.4%</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  FILE NOT FOUND GUARD
# ─────────────────────────────────────────────────────────────
if not DATA_LOADED:
    st.error(f"""
    **Data or model file not found.**  
    Please ensure `data.csv` and `GB.pkl` are in the **same directory** as `app.py`, then restart.

    Missing: `{missing}`
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────
#  PRE-COMPUTE GLOBAL STATS
# ─────────────────────────────────────────────────────────────
global_means = df[KEY_BEHAVIORAL].mean().to_dict()
global_stds  = df[KEY_BEHAVIORAL].std().to_dict()
for col in KEY_BEHAVIORAL:
    global_means[f"_std_{col}"] = global_stds.get(col, 1)

# Identify model feature columns (exclude meta cols)
META_COLS = {"id_student"}
model_features = [c for c in df.columns if c not in META_COLS and c != "study_method_preference"]

# ═══════════════════════════════════════════════════════════════
#  MODE 1 — EXISTING STUDENTS
# ═══════════════════════════════════════════════════════════════
if "Dashboard" in mode:

    # ── KPI Row ───────────────────────────────────────────────
    total = len(df)
    if "engagement_classification" in df.columns:
        risk_counts = df["engagement_classification"].value_counts()
        # 2 = Low engagement = High burnout risk
        n_high   = int(risk_counts.get(2, 0))
        n_mod    = int(risk_counts.get(0, 0))
        n_low    = int(risk_counts.get(1, 0))
    else:
        n_high, n_mod, n_low = 0, 0, 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Students", f"{total:,}")
    with k2:
        st.metric("🚨 High Risk", f"{n_high:,}", delta=f"{n_high/total*100:.1f}% of cohort",
                  delta_color="inverse")
    with k3:
        st.metric("⚠️ Moderate Risk", f"{n_mod:,}", delta=f"{n_mod/total*100:.1f}% of cohort",
                  delta_color="off")
    with k4:
        st.metric("✅ Low Risk", f"{n_low:,}", delta=f"{n_low/total*100:.1f}% of cohort")

    st.markdown("")

    # ── Overview Charts Row ───────────────────────────────────
    ov1, ov2 = st.columns([1, 2])
    with ov1:
        st.plotly_chart(build_risk_donut(df), use_container_width=True)

    with ov2:
        # Engagement distribution by score
        fig_scatter = px.scatter(
            df.sample(min(2000, len(df))),
            x="score", y="average_engagement",
            color=df.sample(min(2000, len(df)))["engagement_classification"].map(ENGAGEMENT_LABEL),
            color_discrete_map=RISK_COLOR,
            opacity=0.5,
            labels={"score": "Avg Assessment Score", "average_engagement": "Avg Engagement Score"},
            title="Engagement vs. Score (random 2K sample)",
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(**PLOTLY_THEME, height=260)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Student Selector ─────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Individual Student Analysis</div>', unsafe_allow_html=True)

    col_search, col_info = st.columns([2, 3])
    with col_search:
        if "id_student" in df.columns:
            student_ids = sorted(df["id_student"].unique().tolist())
            selected_id = st.selectbox(
                "Select Student ID",
                options=student_ids,
                format_func=lambda x: f"Student {x}",
            )
        else:
            selected_id = st.number_input("Enter Student Index", 0, len(df)-1, 0)

    if "id_student" in df.columns:
        student_row_full = df[df["id_student"] == selected_id].iloc[0]
    else:
        student_row_full = df.iloc[int(selected_id)]

    # ── Prediction ───────────────────────────────────────────
    feature_row = student_row_full.drop(labels=[c for c in ["id_student","study_method_preference"] if c in student_row_full.index], errors="ignore")
    X_student   = feature_row.values.reshape(1, -1)

    try:
        pred_class   = model.predict(X_student)[0]
        pred_proba   = model.predict_proba(X_student)
        risk_label   = get_risk_label(pred_class)
        risk_score   = calculate_risk_score(pred_proba)

        # override thresholds from sidebar
        if   risk_score < t1: risk_label = "Low Risk"
        elif risk_score < t2: risk_label = "Moderate Risk"
        else:                 risk_label = "High Risk"

    except Exception as ex:
        st.error(f"Prediction error: {ex}")
        st.stop()

    risk_color = RISK_COLOR[risk_label]
    badge_cls  = RISK_BADGE[risk_label]

    with col_info:
        st.markdown(f"""
        <div class="risk-card" style="border-left: 4px solid {risk_color}; margin-top:1.7rem;">
            <span class="badge {badge_cls}">{risk_label}</span>
            <span style="font-size:0.82rem; color:#64748B; margin-left:0.8rem;">
                Student&nbsp;<span class="mono">{selected_id}</span>
            </span>
            <div style="font-size:0.85rem; color:#94A3B8; margin-top:0.5rem;">
                {INTERVENTIONS[risk_label]['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Main Analysis Tabs ────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Risk Profile", "📊  Behavioral Analysis",
        "🎯  Radar & Trends", "💡  Interventions"
    ])

    with tab1:
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(build_gauge(risk_score, risk_label), use_container_width=True)

        with g2:
            # Probability breakdown
            class_names  = [ENGAGEMENT_LABEL.get(i, str(i)) for i in range(len(pred_proba[0]))]
            class_colors = [RISK_COLOR.get(n, "#94A3B8") for n in class_names]
            fig_prob = go.Figure(go.Bar(
                x=class_names, y=pred_proba[0] * 100,
                marker_color=class_colors,
                text=[f"{p*100:.1f}%" for p in pred_proba[0]],
                textposition="outside",
                textfont=dict(size=12, color="#94A3B8"),
                hovertemplate="%{x}<br>Probability: %{y:.1f}%<extra></extra>",
            ))
            fig_prob.update_layout(
                **PLOTLY_THEME, height=260,
                title=dict(text="Class Probabilities", font=dict(size=13, color="#94A3B8")),
                yaxis_range=[0, 115],
                yaxis_gridcolor="#1F2937"
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Key stats inline
        st.markdown("")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Score",          f"{student_row_full.get('score', 0):.1f}")
        m2.metric("VLE Clicks",     f"{int(student_row_full.get('sum', 0)):,}")
        m3.metric("Submissions",    f"{int(student_row_full.get('count', 0)):,}")
        m4.metric("Act. Diversity", f"{student_row_full.get('activity_diversity', 0):.2f}")
        m5.metric("Improvement",    f"{student_row_full.get('improvement_rate', 0):+.3f}")

    with tab2:
        deviations = get_deviation_data(student_row_full, global_means)
        st.plotly_chart(build_deviation_chart(deviations), use_container_width=True)

        # Two distribution charts side by side
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                build_cohort_distribution(df, student_row_full.get("score", 0), "score"),
                use_container_width=True
            )
        with d2:
            st.plotly_chart(
                build_cohort_distribution(df, student_row_full.get("sum", 0), "sum"),
                use_container_width=True
            )

        d3, d4 = st.columns(2)
        with d3:
            st.plotly_chart(
                build_cohort_distribution(df, student_row_full.get("module_engagement_rate", 0), "module_engagement_rate"),
                use_container_width=True
            )
        with d4:
            st.plotly_chart(
                build_cohort_distribution(df, student_row_full.get("submission_timeliness", 0), "submission_timeliness"),
                use_container_width=True
            )

    with tab3:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(
                build_radar_chart(student_row_full.to_dict(), global_means),
                use_container_width=True
            )
        with r2:
            fig_trend = build_score_history_chart(df, selected_id)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)

        # Engagement breakdown table
        with st.expander("📋 Full Feature Snapshot", expanded=False):
            snap_cols = KEY_BEHAVIORAL + ["average_engagement", "kmeans_cluster"]
            snap_data = {
                "Feature":        [BEHAVIORAL_LABELS.get(c, c) for c in snap_cols if c in student_row_full],
                "Student Value":  [round(float(student_row_full[c]), 4) for c in snap_cols if c in student_row_full],
                "Cohort Mean":    [round(float(global_means.get(c, 0)), 4) for c in snap_cols if c in student_row_full],
                "Z-score":        [round(float((student_row_full[c] - global_means.get(c, 0)) /
                                               max(global_stds.get(c, 1), 1e-9)), 3)
                                   for c in snap_cols if c in student_row_full],
            }
            st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)

    with tab4:
        intervention = INTERVENTIONS[risk_label]
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
            <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; color:{risk_color};">
                {intervention['level_label']}
            </div>
            <span class="badge {RISK_BADGE[risk_label]}">{risk_label}</span>
        </div>
        <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:1.5rem; line-height:1.6;">
            {intervention['summary']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-family:\'DM Serif Display\',serif; font-size:1.1rem; color:#F0F4F8; margin-bottom:0.75rem;">Recommended Actions</div>', unsafe_allow_html=True)
        for action in intervention["actions"]:
            st.markdown(f"""
            <div class="intervention-card {intervention['badge']}">
                {action}
            </div>
            """, unsafe_allow_html=True)

        # Weakest behavioural signals
        st.markdown("")
        st.markdown('<div style="font-family:\'DM Serif Display\',serif; font-size:1.1rem; color:#F0F4F8; margin-bottom:0.75rem;">Critical Metrics Below Average</div>', unsafe_allow_html=True)
        deviations = get_deviation_data(student_row_full, global_means)
        weak_feats = sorted(
            [(k, v) for k, v in deviations.items() if v["z_score"] < -0.5],
            key=lambda x: x[1]["z_score"]
        )
        if weak_feats:
            for feat, data in weak_feats[:4]:
                z = data["z_score"]
                st.markdown(f"""
                <div style="background:#1C2333; border:1px solid #2D3748; border-radius:8px;
                            padding:0.75rem 1rem; margin:0.35rem 0; display:flex;
                            justify-content:space-between; align-items:center;">
                    <span style="font-size:0.88rem; color:#94A3B8;">{data['label']}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#EF4444;">
                        {z:+.2f}σ &nbsp;|&nbsp; {data['student']:.3f} vs {data['global']:.3f} avg
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#10B981; font-size:0.9rem;">✅ No metrics significantly below cohort average.</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MODE 2 — SIMULATE NEW STUDENT
# ═══════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">🧪 Simulate a New Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.88rem; color:#64748B; margin-bottom:1.5rem;">Adjust the sliders below to define a hypothetical student. The model will instantly re-calculate their burnout risk based on the configured behavioral indicators.</div>', unsafe_allow_html=True)

    # ── Build median profile ──────────────────────────────────
    median_profile = df.drop(columns=[c for c in ["id_student","study_method_preference"] if c in df.columns], errors="ignore").median()

    # ── Input Form ────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div style="font-size:0.75rem; color:#64748B; letter-spacing:0.08em; text-transform:uppercase; font-weight:600; margin-bottom:0.75rem;">Academic Performance</div>', unsafe_allow_html=True)

        avg_score = st.slider(
            "Average Assessment Score (0–100)",
            0.0, 100.0,
            float(median_profile.get("score", 75.0)), 0.5,
            help="Mean score across all assessments for this student."
        )
        submission_delay = st.slider(
            "Submission Timeliness (days; negative = early)",
            -30.0, 60.0,
            float(median_profile.get("submission_timeliness", -2.0)), 0.5,
            help="Average days late (positive) or early (negative) when submitting."
        )
        improvement = st.slider(
            "Score Improvement Rate",
            -1.0, 1.0,
            float(median_profile.get("improvement_rate", 0.0)), 0.01,
            help="Average per-assessment score change over the semester."
        )

    with c2:
        st.markdown('<div style="font-size:0.75rem; color:#64748B; letter-spacing:0.08em; text-transform:uppercase; font-weight:600; margin-bottom:0.75rem;">Engagement & VLE Activity</div>', unsafe_allow_html=True)

        total_clicks = st.slider(
            "Total VLE Clicks (normalised 0–1)",
            0.0, 1.0,
            float(median_profile.get("sum", 0.05)), 0.01,
            help="Normalised total interaction count with VLE materials."
        )
        session_count = st.slider(
            "Interaction Sessions (normalised 0–1)",
            0.0, 1.0,
            float(median_profile.get("count", 0.05)), 0.01,
            help="Normalised number of distinct login/activity sessions."
        )
        engagement_rate = st.slider(
            "Module Engagement Rate (normalised 0–1)",
            0.0, 1.0,
            float(median_profile.get("module_engagement_rate", 0.05)), 0.01,
            help="Daily VLE engagement intensity (clicks / module length)."
        )
        diversity = st.slider(
            "Activity Diversity (normalised 0–1)",
            0.0, 1.0,
            float(median_profile.get("activity_diversity", 0.5)), 0.01,
            help="Proportion of different VLE activity types used."
        )

    st.markdown("")

    with st.expander("⚙️  Advanced / Demographic Overrides", expanded=False):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            gender = st.selectbox("Gender", ["M", "F"])
            disability = st.selectbox("Disability", ["N", "Y"])
        with ac2:
            education = st.selectbox("Education Level", [
                "A Level or Equivalent", "HE Qualification", "Lower Than A Level"
            ])
            age_band = st.selectbox("Age Band", ["0-35", "35+"])
        with ac3:
            imd_band = st.selectbox("IMD Band", [
                "10-20","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%","0-10%"
            ])
            prev_attempts = st.slider("Previous Attempts", 0, 6, 0)

    run_sim = st.button("🔮  Calculate Risk Score", use_container_width=False)
    st.markdown("")

    if run_sim or True:  # auto-run on slider change
        # Build simulated profile from median
        sim = median_profile.copy()

        sim["score"]                    = avg_score
        sim["sum"]                      = total_clicks
        sim["count"]                    = session_count
        sim["submission_timeliness"]    = submission_delay
        sim["module_engagement_rate"]   = engagement_rate
        sim["activity_diversity"]       = diversity
        sim["improvement_rate"]         = improvement
        sim["num_of_prev_attempts"]     = prev_attempts

        # One-hot gender
        sim["gender_M"] = 1.0 if gender == "M" else 0.0
        sim["gender_F"] = 1.0 if gender == "F" else 0.0

        # One-hot disability
        sim["disability_N"] = 1.0 if disability == "N" else 0.0
        sim["disability_Y"] = 1.0 if disability == "Y" else 0.0

        # One-hot education
        for ed in ["A Level or Equivalent","HE Qualification","Lower Than A Level"]:
            sim[f"highest_education_{ed}"] = 1.0 if education == ed else 0.0

        # One-hot age
        for ab in ["0-35","35+"]:
            sim[f"age_band_{ab}"] = 1.0 if age_band == ab else 0.0

        # One-hot IMD
        for band in ["0-10%","10-20","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]:
            sim[f"imd_band_{band}"] = 1.0 if imd_band == band else 0.0

        # Derive engineered features
        sim["assessment_engagement_score"] = total_clicks * session_count
        sim["weighted_engagement"]         = sim["assessment_engagement_score"] * float(median_profile.get("weight", 10))
        sim["engagement_consistency"]      = abs(improvement) * 0.3
        sim["average_engagement"]          = np.mean([total_clicks, session_count, engagement_rate, diversity])
        sim["engagement_classification"]   = 0

        X_sim = sim.values.reshape(1, -1)

        try:
            pred_class_sim  = model.predict(X_sim)[0]
            pred_proba_sim  = model.predict_proba(X_sim)
            risk_label_sim  = get_risk_label(pred_class_sim)
            risk_score_sim  = calculate_risk_score(pred_proba_sim)

            if   risk_score_sim < t1: risk_label_sim = "Low Risk"
            elif risk_score_sim < t2: risk_label_sim = "Moderate Risk"
            else:                     risk_label_sim = "High Risk"

        except Exception as ex:
            st.error(f"Simulation prediction error: {ex}. Ensure GB.pkl feature order matches the data.")
            st.stop()

        risk_color_sim = RISK_COLOR[risk_label_sim]

        # ── Sim Results ───────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Simulation Results</div>', unsafe_allow_html=True)

        res1, res2, res3 = st.columns(3)

        with res1:
            st.plotly_chart(build_gauge(risk_score_sim, risk_label_sim), use_container_width=True)

        with res2:
            # Probability bar
            class_names_s  = [ENGAGEMENT_LABEL.get(i, str(i)) for i in range(len(pred_proba_sim[0]))]
            class_colors_s = [RISK_COLOR.get(n, "#94A3B8") for n in class_names_s]
            fig_p = go.Figure(go.Bar(
                x=class_names_s, y=pred_proba_sim[0] * 100,
                marker_color=class_colors_s,
                text=[f"{p*100:.1f}%" for p in pred_proba_sim[0]],
                textposition="outside",
                textfont=dict(size=12, color="#94A3B8"),
            ))
            fig_p.update_layout(**PLOTLY_THEME, height=260,
                                title=dict(text="Predicted Probabilities", font=dict(size=13, color="#94A3B8")),
                                yaxis_range=[0, 115],
                                yaxis_gridcolor="#1F2937")
            st.plotly_chart(fig_p, use_container_width=True)

        with res3:
            intervention_sim = INTERVENTIONS[risk_label_sim]
            st.markdown(f"""
            <div class="risk-card" style="border-left:4px solid {risk_color_sim}; height:200px; overflow:auto;">
                <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:{risk_color_sim}; margin-bottom:0.5rem;">
                    {intervention_sim['level_label']}
                </div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.6;">
                    {intervention_sim['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Simulation inputs summary
        sim_tab1, sim_tab2 = st.tabs(["💡  Interventions", "📊  Input Summary"])
        with sim_tab1:
            for action in INTERVENTIONS[risk_label_sim]["actions"]:
                st.markdown(f"""
                <div class="intervention-card {INTERVENTIONS[risk_label_sim]['badge']}">
                    {action}
                </div>
                """, unsafe_allow_html=True)

        with sim_tab2:
            summary_df = pd.DataFrame({
                "Metric": [
                    "Avg Score", "Submission Timeliness (days)", "Improvement Rate",
                    "VLE Clicks (norm.)", "Sessions (norm.)", "Engagement Rate (norm.)",
                    "Activity Diversity (norm.)", "Previous Attempts",
                ],
                "Simulated Value": [
                    f"{avg_score:.1f}", f"{submission_delay:+.1f}", f"{improvement:+.3f}",
                    f"{total_clicks:.3f}", f"{session_count:.3f}", f"{engagement_rate:.3f}",
                    f"{diversity:.3f}", str(prev_attempts),
                ],
                "Cohort Mean": [
                    f"{global_means.get('score',0):.1f}",
                    f"{global_means.get('submission_timeliness',0):+.1f}",
                    f"{global_means.get('improvement_rate',0):+.3f}",
                    f"{global_means.get('sum',0):.3f}",
                    f"{global_means.get('count',0):.3f}",
                    f"{global_means.get('module_engagement_rate',0):.3f}",
                    f"{global_means.get('activity_diversity',0):.3f}",
                    f"{global_means.get('num_of_prev_attempts',0):.1f}",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid #1F2937;
            text-align:center; font-size:0.75rem; color:#374151; letter-spacing:0.04em;">
    OULAD Burnout Risk Dashboard &nbsp;·&nbsp; Gradient Boosting (93.4% accuracy)
    &nbsp;·&nbsp; Open University Learning Analytics Dataset
    &nbsp;·&nbsp; Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)