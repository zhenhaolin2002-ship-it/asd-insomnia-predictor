"""
ASD 人群中重度失眠风险预测工具
基于 Logistic Regression + SHAP 解释
所有输入范围均来自真实数据集
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Insomnia Risk Predictor",
    page_icon="🌙",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# ⚠️ REDUCED-MODEL (6-INPUT / 12-MEDIAN-IMPUTED) PERFORMANCE
# ─────────────────────────────────────────────────────────────
# These MUST be computed on your held-out test / external validation set
# using the SAME deployment condition as this app: the 6 sidebar features
# take their real values, and the other 12 LASSO_FEATURES are forced to
# FEATURE_MEDIANS below (exactly as run_prediction() does). Do NOT reuse
# the full-model (all-18-real-features) metrics from the manuscript here —
# they will not match what a website user actually experiences, and this
# is precisely the gap the reviewer flagged.
#
# Recommended workflow to fill these in:
#   1. Take your test/validation set.
#   2. For each row, keep only the 6 sidebar variables; overwrite the
#      other 12 columns with FEATURE_MEDIANS.
#   3. Run predict_proba() through the saved pipeline.
#   4. Compute AUC (+ 95% CI via bootstrap), calibration slope/intercept
#      (or a calibration plot), and sensitivity/specificity/PPV/NPV at
#      your chosen threshold (e.g., Youden's J, or a clinically motivated
#      cutoff prioritizing sensitivity given the vulnerable population).
#
# Replace every None below with your actual numbers before re-deploying.
REDUCED_MODEL_METRICS = {
    "validation_set":        "Internal test set (n = 293, 30% held-out split)",
    "auc":                   0.711,
    "auc_ci":                "0.649–0.772",
    "brier_score":           0.196,
    "calibration_slope":     0.979,
    "calibration_intercept": -0.664,
    "threshold":             0.30,
    "sensitivity":           0.887,
    "specificity":           0.423,
    "ppv":                   0.366,
    "npv":                   0.909,
}

def _fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"

# ─────────────────────────────────────────────────────────────
# 特征配置
# ─────────────────────────────────────────────────────────────
LASSO_FEATURES = [
    "hobby_no_hobby", "gender", "diagnosis_anxiety", "hobby_sports_exercise",
    "trauma_cyberbullying", "trauma_sexual_assault", "own_bedroom",
    "hobby_category_count", "physical_activity_level", "gad9_no_sleep_item5_total",
    "phq8_no_sleep_total", "household_car_count", "trauma_count",
    "family_atmosphere", "residence", "age", "screen_short_video_pct", "bmi",
]

SHORT_LABELS = {
    "hobby_no_hobby":            "No Hobby",
    "gender":                    "Gender",
    "diagnosis_anxiety":         "Anxiety Dx",
    "hobby_sports_exercise":     "Sports Hobby",
    "trauma_cyberbullying":      "Cyberbully",
    "trauma_sexual_assault":     "Sexual Assault",
    "own_bedroom":               "Own Bedroom",
    "hobby_category_count":      "Hobby Count",
    "physical_activity_level":   "PA Level",
    "gad9_no_sleep_item5_total": "GAD-9",
    "phq8_no_sleep_total":       "PHQ-8",
    "household_car_count":       "Cars",
    "trauma_count":              "Trauma Count",
    "family_atmosphere":         "Family Atmos.",
    "residence":                 "Residence",
    "age":                       "Age",
    "screen_short_video_pct":    "Short Video %",
    "bmi":                       "BMI",
}
FEAT_NAMES = [SHORT_LABELS[f] for f in LASSO_FEATURES]

# 各特征真实数据中位数（用于未输入特征的默认值）
FEATURE_MEDIANS = {
    "hobby_no_hobby":            0,
    "gender":                    2,
    "diagnosis_anxiety":         0,
    "hobby_sports_exercise":     0,
    "trauma_cyberbullying":      0,
    "trauma_sexual_assault":     0,
    "own_bedroom":               1,
    "hobby_category_count":      2,
    "physical_activity_level":   0,
    "gad9_no_sleep_item5_total": 15,
    "phq8_no_sleep_total":       12,
    "household_car_count":       1,
    "trauma_count":              2,
    "family_atmosphere":         2,
    "residence":                 2,
    "age":                       23,
    "screen_short_video_pct":    10.0,
    "bmi":                       21.48,
}

# ─────────────────────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import glob
    matches = glob.glob("Best_Model_*.joblib")
    if not matches:
        return None
    return joblib.load(matches[0])

pipeline = load_model()

if pipeline is None:
    st.error(
        "⚠️ Model file not found. "
        "Please upload `Best_Model_*.joblib` to the same folder as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# 标题 + 非诊断免责声明（顶部，始终可见）
# ─────────────────────────────────────────────────────────────
st.title("🌙 Sleep Difficulty Risk Predictor in ASD")
st.markdown(
    "Enter participant information in the **sidebar**. "
    "The model will estimate the risk of **moderate-to-severe insomnia** "
    "in individuals with **Autism Spectrum Disorder (ASD)**, "
    "and explain key contributing factors via SHAP."
)

st.warning(
    "**⚠️ For research and educational use only.** This tool is not a "
    "diagnostic device and should not replace professional clinical "
    "judgment. Please consult a qualified clinician for any sleep, mood, "
    "or anxiety concerns."
)

with st.expander("📈 Reduced-model performance (6 sidebar inputs, remaining features at cohort median)", expanded=True):
    m = REDUCED_MODEL_METRICS
    st.caption(
        "Metrics reflect this tool's actual input mode (6 sidebar variables "
        "entered, remaining 12 fixed at population medians), evaluated on **"
        + (m["validation_set"] or "not yet reported") + "**."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", _fmt(m["auc"]), help=f"95% CI: {_fmt(m['auc_ci'])}")
    c2.metric("Brier score", _fmt(m["brier_score"]))
    c3.metric("Calibration slope", _fmt(m["calibration_slope"]))
    c4.metric("Calibration intercept", _fmt(m["calibration_intercept"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sensitivity", _fmt(m["sensitivity"]))
    c6.metric("Specificity", _fmt(m["specificity"]))
    c7.metric("PPV", _fmt(m["ppv"]))
    c8.metric("NPV", _fmt(m["npv"]))

    st.caption(
        f"Operating threshold for risk classification below: "
        f"**{m['threshold']:.2f}** predicted probability "
        "(sensitivity/specificity above are reported at this threshold)."
    )
st.divider()

# ─────────────────────────────────────────────────────────────
# 侧边栏输入（Top-6 特征，范围来自真实数据）
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Participant Information")
    st.caption(
        "Fill in the 6 key predictors. "
        "Other features use population median values."
    )

    # ── GAD-9（去除睡眠条目，范围 0–36）
    gad9 = st.number_input(
        "GAD-9 Score (excl. sleep item)",
        min_value=0, max_value=36, value=15, step=1,
        help="Total GAD-9 score excluding the sleep-related item (range: 0–36)."
    )

    # ── PHQ-8（去除睡眠条目，范围 0–24）
    phq8 = st.number_input(
        "PHQ-8 Score (excl. sleep item)",
        min_value=0, max_value=24, value=12, step=1,
        help="Total PHQ-8 score excluding the sleep-related item (range: 0–24)."
    )

    # ── 焦虑诊断
    anxiety = st.selectbox(
        "Anxiety Diagnosis",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Has the participant been diagnosed with an anxiety disorder?"
    )

    # ── 短视频使用百分比（0–99）
    short_video = st.slider(
        "Short Video Screen Time (%)",
        min_value=0.0, max_value=99.0, value=10.0, step=0.5,
        help="Percentage of daily screen time spent on short-form videos (0–99%)."
    )

    # ── 年龄（18–53）
    age = st.number_input(
        "Age (years)",
        min_value=18, max_value=53, value=23, step=1,
    )

    # ── 性别
    gender = st.selectbox(
        "Gender",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female",
    )

    st.divider()
    predict_btn = st.button(
        "🔍 Predict Risk",
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────────────────────
# 构建完整输入向量
# ─────────────────────────────────────────────────────────────
input_values = {**FEATURE_MEDIANS}   # 先填中位数
input_values["gad9_no_sleep_item5_total"] = gad9
input_values["phq8_no_sleep_total"]       = phq8
input_values["diagnosis_anxiety"]         = anxiety
input_values["screen_short_video_pct"]    = short_video
input_values["age"]                       = age
input_values["gender"]                    = gender

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def run_prediction(vals):
    row   = pd.DataFrame([{f: vals[f] for f in LASSO_FEATURES}])
    prob  = float(pipeline.predict_proba(row)[0, 1])

    imputer = pipeline.named_steps["imputer"]
    scaler  = pipeline.named_steps["scaler"]
    clf     = pipeline.named_steps["clf"]
    row_t   = scaler.transform(imputer.transform(row))

    bg         = np.zeros((1, row_t.shape[1]))
    explainer  = shap.LinearExplainer(
        clf, bg, feature_perturbation="interventional"
    )
    shap_vals  = explainer.shap_values(row_t)[0]
    base_val   = float(explainer.expected_value)
    return prob, shap_vals, base_val, row_t[0]


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180,
                bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def plot_waterfall(shap_vals, base_val, row_t, max_display=12):
    expl = shap.Explanation(
        values        = shap_vals,
        base_values   = base_val,
        data          = row_t,
        feature_names = FEAT_NAMES,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.sca(ax)
    shap.waterfall_plot(expl, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    plt.gca().set_xlabel("SHAP value", fontsize=12)
    plt.tight_layout()
    return fig


def plot_force(shap_vals, base_val, row_t, prob):
    RED  = "#E83B3B"
    BLUE = "#2979FF"

    top_n     = 10
    order     = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
    pos_feats = [(FEAT_NAMES[i], shap_vals[i], row_t[i])
                 for i in order if shap_vals[i] > 0]
    neg_feats = [(FEAT_NAMES[i], shap_vals[i], row_t[i])
                 for i in order if shap_vals[i] <= 0]

    fig, ax = plt.subplots(figsize=(13, 2.0))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    H, y0, OVL = 0.55, 0.225, 0.008
    pos_abs = [abs(s) for _, s, _ in pos_feats]
    neg_abs = [abs(s) for _, s, _ in neg_feats]
    total   = sum(pos_abs) + sum(neg_abs) + 1e-9
    pos_ws  = [v / total for v in pos_abs]
    neg_ws  = [v / total for v in neg_abs]

    def arrow_block(ax, x0, w, label, color):
        tip = min(OVL * 3, w * 0.3)
        ax.fill_betweenx([y0, y0 + H], x0, x0 + w - tip,
                         color=color, alpha=0.92, linewidth=0)
        ax.fill([x0+w-tip, x0+w, x0+w-tip],
                [y0, y0+H/2, y0+H],
                color=color, alpha=0.92, linewidth=0)
        cx = x0 + (w - tip) / 2
        fs = 8.0 if w > 0.07 else (6.5 if w > 0.04 else 0)
        if fs > 0:
            ax.text(cx, y0+H/2, label, ha="center", va="center",
                    fontsize=fs, color="white", fontweight="bold",
                    clip_on=True)

    x_cur = 0.0
    for (fn, sv, fv), w in zip(pos_feats, pos_ws):
        arrow_block(ax, x_cur, max(w, 0.01), f"{fn}={fv:.2g}", RED)
        x_cur += max(w, 0.01) - OVL

    sep = x_cur + OVL
    ax.axvline(sep, color="#111", linewidth=2.2, zorder=5)
    ax.text(sep, y0+H+0.14, f"Risk = {prob:.3f}",
            ha="center", fontsize=10, fontweight="bold", color="#111")

    x_cur = sep
    for (fn, sv, fv), w in zip(neg_feats, neg_ws):
        arrow_block(ax, x_cur, max(w, 0.01), f"{fn}={fv:.2g}", BLUE)
        x_cur += max(w, 0.01) - OVL

    ax.axvline(1.0, color="#999", linewidth=1.2, linestyle="--", zorder=4)
    ax.text(1.0, y0-0.22, f"Base\n{base_val:.3f}",
            ha="center", fontsize=8, color="#666")

    red_p  = mpatches.Patch(color=RED,  label="Risk-increasing ↑")
    blue_p = mpatches.Patch(color=BLUE, label="Risk-decreasing ↓")
    ax.legend(handles=[red_p, blue_p], fontsize=9,
              loc="upper right", framealpha=0.9,
              bbox_to_anchor=(1.0, 1.65))
    ax.set_xlim(-0.01, 1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Computing prediction and SHAP values…"):
        prob, shap_vals, base_val, row_t = run_prediction(input_values)

    THRESH = REDUCED_MODEL_METRICS["threshold"]

    # ── 风险结果区 ────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.metric("Predicted Risk Probability", f"{prob:.1%}")
        if prob >= THRESH:
            st.error(f"⚠️ **Above the {THRESH:.0%} threshold** — flagged as elevated risk.")
        else:
            st.success(f"✅ **Below the {THRESH:.0%} threshold** — not flagged as elevated risk.")
        st.caption("A research estimate, not a diagnosis — see performance metrics above.")

        st.markdown("**Top-3 driving factors:**")
        top3 = np.argsort(np.abs(shap_vals))[-3:][::-1]
        for i in top3:
            arrow = "↑ increases" if shap_vals[i] > 0 else "↓ decreases"
            st.markdown(
                f"- **{FEAT_NAMES[i]}** {arrow} risk "
                f"*(SHAP = {shap_vals[i]:+.3f})*"
            )

    with col_right:
        # 进度条风险仪表
        fig_g, ax_g = plt.subplots(figsize=(6, 1.4))
        ax_g.barh(0, 1.0, color="#eeeeee", height=0.5)
        bar_color = "#E83B3B" if prob >= THRESH else "#27AE60"
        ax_g.barh(0, prob, color=bar_color, height=0.5)
        ax_g.axvline(THRESH, color="#555555", linewidth=1.5, linestyle="--")
        ax_g.text(
            min(prob + 0.02, 0.95), 0,
            f"{prob:.1%}",
            va="center", fontsize=14, fontweight="bold", color=bar_color
        )
        ax_g.text(THRESH, -0.42, f"{THRESH:.0%} threshold",
                  ha="center", fontsize=9, color="#555")
        ax_g.set_xlim(0, 1)
        ax_g.set_yticks([])
        ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_g.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
        for sp in ["top", "left", "right"]:
            ax_g.spines[sp].set_visible(False)
        ax_g.set_title("Predicted Insomnia Risk", fontsize=12, pad=8)
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    st.divider()

    # ── SHAP 图 Tab ───────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Waterfall Plot", "⚡ Force Plot"])

    with tab1:
        st.markdown(
            "Each bar shows how much a feature **pushes the prediction** "
            "away from the baseline value."
        )
        fig_wf = plot_waterfall(shap_vals, base_val, row_t)
        st.pyplot(fig_wf, use_container_width=True)
        st.download_button(
            "⬇️ Download Waterfall Plot (PNG)",
            data=fig_to_bytes(fig_wf),
            file_name="waterfall_plot.png",
            mime="image/png",
        )
        plt.close(fig_wf)

    with tab2:
        st.markdown(
            "**Red** blocks increase insomnia risk · "
            "**Blue** blocks decrease insomnia risk"
        )
        fig_fp = plot_force(shap_vals, base_val, row_t, prob)
        st.pyplot(fig_fp, use_container_width=True)
        st.download_button(
            "⬇️ Download Force Plot (PNG)",
            data=fig_to_bytes(fig_fp),
            file_name="force_plot.png",
            mime="image/png",
        )
        plt.close(fig_fp)

    st.divider()

    # ── 完整 SHAP 数值表 ──────────────────────────────────────
    with st.expander("📋 View Full SHAP Values Table"):
        df_shap = pd.DataFrame({
            "Feature":     FEAT_NAMES,
            "Input Value": [input_values[f] for f in LASSO_FEATURES],
            "SHAP Value":  np.round(shap_vals, 4),
            "Direction":   ["↑ Risk" if v > 0 else "↓ Risk" for v in shap_vals],
        }).sort_values("SHAP Value", key=abs, ascending=False).reset_index(drop=True)
        st.dataframe(df_shap, use_container_width=True)

    st.caption("For research/educational use only — not a substitute for clinical assessment.")

else:
    # 未点击时的说明页
    st.info(
        "👈 Fill in the participant's information in the **left sidebar**, "
        "then click **Predict Risk**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### How it works
1. Enter **6 key predictors** in the sidebar
2. Click **🔍 Predict Risk**
3. View predicted probability + SHAP explanation
4. Download plots if needed
        """)
    with col2:
        st.markdown("""
### Input Variables
| Variable | Range |
|----------|-------|
| GAD-9 Score | 0 – 36 |
| PHQ-8 Score | 0 – 24 |
| Anxiety Diagnosis | Yes / No |
| Short Video % | 0 – 99% |
| Age | 18 – 53 |
| Gender | Male / Female |
        """)

    m = REDUCED_MODEL_METRICS
    st.markdown(f"""
---
### About the model
- **Algorithm**: Logistic Regression with LASSO feature selection
- **Target**: Moderate-to-severe insomnia (ISI ≥ 15)
- **Population**: Adults with **Autism Spectrum Disorder (ASD)**, n = 976 (27.2% prevalence)
- **Remaining 12 features** are fixed at population median values
- **This deployment's performance**: AUC {_fmt(m['auc'])} (95% CI {_fmt(m['auc_ci'])}),
  sensitivity {_fmt(m['sensitivity'])} / specificity {_fmt(m['specificity'])} at a {m['threshold']:.0%} threshold
  — see the panel above for full detail.

For research and educational use only; not a substitute for clinical assessment.
    """)
