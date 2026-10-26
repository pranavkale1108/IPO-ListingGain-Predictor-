"""
streamlit_app.py
----------------
Interactive Web Application for the IPO Listing-Gain Predictor (NSE/BSE).

Provides:
- Input form with sensible defaults and preset test IPO loader.
- Real-time predictive intelligence:
  1. Expected Listing Gain % (Gradient Boosting Regressor)
  2. Positive Listing Probability % (Gradient Boosting Classifier)
  3. Investment Verdict: Apply / Neutral / Avoid (Support Vector Machine)
  4. Top-5 Most Similar Historical Peers (k-NN Engine)
  5. Decoded Market Mood Banner (Hidden Markov Model)
  6. Cluster Archetype Profile (K-Means)
- Model comparison leaderboard and ablation study.
- Mandatory SEBI educational disclaimer.
"""

from typing import Dict, Any, Optional
import sys
import os

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.predict import predict_ipo, get_market_mood_for_date, load_artifacts

# ---------------------------------------------------------
# Page Configuration & Modern Design System
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPO Listing-Gain Predictor | NSE/BSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern financial terminal aesthetics
st.markdown(
    """
    <style>
    /* Global Typography & Background Elements */
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Fjalla+One&family=Gruppo&family=Montserrat:ital,wght@0,100..900;1,100..900&family=Outfit:wght@100..900&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText, p, span, label, input, select, button, div {
        font-family: 'Outfit', 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', 'Outfit', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    .metric-value-huge {
        font-family: 'Outfit', 'Montserrat', sans-serif !important;
        font-weight: 800;
    }
    
    /* Disclaimer Banner */
    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(245, 158, 11, 0.12));
        border-left: 4px solid #ef4444;
        padding: 12px 18px;
        border-radius: 8px;
        color: #f87171;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    
    /* Market Mood Banner */
    .mood-banner-hot {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(249, 115, 22, 0.15));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 14px 20px;
        color: #f87171;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    .mood-banner-warm {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(234, 179, 8, 0.15));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 14px 20px;
        color: #fbbf24;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    .mood-banner-cold {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(14, 165, 233, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 10px;
        padding: 14px 20px;
        color: #60a5fa;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .metric-value-huge {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    /* Verdict Badges */
    .badge-apply {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-neutral {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-avoid {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Educational Disclaimer Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚖️ <strong>EDUCATIONAL PROJECT ONLY — NOT FINANCIAL OR INVESTMENT ADVICE:</strong> 
        This application is an academic machine learning mini-project for the Indian equity market (NSE/BSE). 
        Predictions are generated by statistical ML models and should NEVER be utilized as investment recommendations. 
        Consult a SEBI-registered investment advisor before making financial decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Historical & Test References
# ---------------------------------------------------------
@st.cache_data
def get_reference_data():
    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    mood_path = os.path.join(PROJECT_ROOT, "data", "processed", "monthly_market_mood.csv")
    metrics_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.csv")
    ablation_path = os.path.join(PROJECT_ROOT, "reports", "ablation_study.csv")

    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    mood_df = pd.read_csv(mood_path) if os.path.exists(mood_path) else pd.DataFrame()
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    ablation_df = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else pd.DataFrame()
    return test_df, mood_df, metrics_df, ablation_df


test_df, mood_df, metrics_df, ablation_df = get_reference_data()

# Pre-defined test cases for 1-click evaluation
PRESET_CASES = {
    "Custom Manual Input": None,
    "Test Case 1 (Apply): Navbharat Consumer Care Ltd (+15.96% actual gain)": {
        "ipo_name": "Navbharat Consumer Care Ltd",
        "listing_date": "2024-01-19",
        "sector": "Chemicals",
        "issue_size_cr": 2711.8,
        "price_band_upper": 105.13,
        "issue_price": 105.13,
        "qib_sub": 44.14,
        "nii_sub": 10.47,
        "retail_sub": 17.77,
        "total_sub": 29.86,
        "gmp_last_day": 35.77,
        "fresh_issue_pct": 43.49,
        "ofs_pct": 56.51,
        "revenue_growth_pct": 47.08,
        "profit_growth_pct": 20.20,
        "pe_vs_peers": 1.44,
        "nifty_7d_trend_pct": -0.04,
        "actual_gain": 15.96,
        "actual_verdict": "Apply",
    },
    "Test Case 2 (Neutral): Synergy Diagnostics Ltd (+2.89% actual gain)": {
        "ipo_name": "Synergy Diagnostics Ltd",
        "listing_date": "2024-01-10",
        "sector": "Consumer Goods",
        "issue_size_cr": 727.45,
        "price_band_upper": 172.38,
        "issue_price": 172.38,
        "qib_sub": 5.40,
        "nii_sub": 30.59,
        "retail_sub": 8.12,
        "total_sub": 10.13,
        "gmp_last_day": 3.23,
        "fresh_issue_pct": 62.31,
        "ofs_pct": 37.69,
        "revenue_growth_pct": 3.81,
        "profit_growth_pct": 12.00,
        "pe_vs_peers": 1.04,
        "nifty_7d_trend_pct": -1.42,
        "actual_gain": 2.89,
        "actual_verdict": "Neutral",
    },
    "Test Case 3 (Avoid): Innova Diagnostics Ltd (-4.29% actual gain)": {
        "ipo_name": "Innova Diagnostics Ltd",
        "listing_date": "2024-01-21",
        "sector": "Chemicals",
        "issue_size_cr": 786.86,
        "price_band_upper": 389.89,
        "issue_price": 389.89,
        "qib_sub": 7.69,
        "nii_sub": 69.04,
        "retail_sub": 12.23,
        "total_sub": 18.48,
        "gmp_last_day": -37.14,
        "fresh_issue_pct": 16.47,
        "ofs_pct": 83.53,
        "revenue_growth_pct": 42.55,
        "profit_growth_pct": 41.78,
        "pe_vs_peers": 1.31,
        "nifty_7d_trend_pct": 4.11,
        "actual_gain": -4.29,
        "actual_verdict": "Avoid",
    },
}

# ---------------------------------------------------------
# Sidebar: IPO Input Form & Preset Selector
# ---------------------------------------------------------
st.sidebar.header("🛠️ IPO Input Configuration")

selected_preset = st.sidebar.selectbox(
    "📂 Quick-Load Test Set IPO:",
    options=list(PRESET_CASES.keys()),
    index=1,  # Default to Test Case 1
    help="Select a real unseen test IPO to automatically populate parameters, or choose Custom to manually test any IPO.",
)

preset_vals = PRESET_CASES.get(selected_preset) or {
    "ipo_name": "Tata NexGen Mobility Ltd",
    "listing_date": "2024-12-15",
    "sector": "Automobile",
    "issue_size_cr": 1250.0,
    "price_band_upper": 450.0,
    "issue_price": 450.0,
    "qib_sub": 35.0,
    "nii_sub": 25.0,
    "retail_sub": 12.0,
    "total_sub": 24.5,
    "gmp_last_day": 85.0,
    "fresh_issue_pct": 70.0,
    "ofs_pct": 30.0,
    "revenue_growth_pct": 28.5,
    "profit_growth_pct": 22.0,
    "pe_vs_peers": 0.95,
    "nifty_7d_trend_pct": 1.25,
}

# Input Form
with st.sidebar.form("ipo_input_form"):
    st.markdown("#### 1. General Company Profile")
    val_name = st.text_input("IPO Company Name", value=preset_vals.get("ipo_name", "Upcoming Tech Debut Ltd"))
    val_date = st.text_input("Tentative Listing Date (YYYY-MM-DD)", value=preset_vals.get("listing_date", "2024-12-15"))
    sectors_list = [
        "Automobile", "Chemicals", "Consumer Goods", "Energy", "Financial Services",
        "Healthcare", "Infrastructure", "Manufacturing", "Retail", "Technology"
    ]
    cur_sector = preset_vals.get("sector", "Technology")
    sec_idx = sectors_list.index(cur_sector) if cur_sector in sectors_list else 0
    val_sector = st.selectbox("Industry Sector", options=sectors_list, index=sec_idx)

    st.markdown("#### 2. Issue Sizing & Pricing")
    col_a, col_b = st.columns(2)
    val_size = col_a.number_input("Issue Size (₹ Cr)", min_value=10.0, max_value=50000.0, value=float(preset_vals.get("issue_size_cr", 1000.0)), step=50.0)
    val_price = col_b.number_input("Issue Price (₹)", min_value=1.0, max_value=10000.0, value=float(preset_vals.get("issue_price", 450.0)), step=10.0)

    st.markdown("#### 3. Subscription Demand Multiples (x)")
    col_q, col_n = st.columns(2)
    val_qib = col_q.number_input("QIB Subscription (x)", min_value=0.0, max_value=500.0, value=float(preset_vals.get("qib_sub", 15.0)), step=1.0)
    val_nii = col_n.number_input("NII Subscription (x)", min_value=0.0, max_value=500.0, value=float(preset_vals.get("nii_sub", 15.0)), step=1.0)

    col_r, col_t = st.columns(2)
    val_retail = col_r.number_input("Retail Subscription (x)", min_value=0.0, max_value=500.0, value=float(preset_vals.get("retail_sub", 8.0)), step=1.0)
    val_total = col_t.number_input("Total Subscription (x)", min_value=0.0, max_value=500.0, value=float(preset_vals.get("total_sub", 14.0)), step=1.0)

    st.markdown("#### 4. Market & Sentiment Signals")
    col_g, col_m = st.columns(2)
    val_gmp = col_g.number_input("Last Day GMP (₹)", value=float(preset_vals.get("gmp_last_day", 45.0)), step=5.0, help="Grey Market Premium in ₹ per share")
    val_nifty = col_m.number_input("Nifty 7D Trend (%)", value=float(preset_vals.get("nifty_7d_trend_pct", 1.0)), step=0.25, help="Nifty 50 7-day trailing percentage return")

    # Mood override option
    mood_options = ["Auto-Detect (HMM)", "Cold", "Warm", "Hot"]
    val_mood_select = st.selectbox("Market Mood Override", options=mood_options, index=0)

    st.markdown("#### 5. Financial & Valuation Fundamentals")
    col_f, col_o = st.columns(2)
    val_fresh = col_f.number_input("Fresh Issue %", min_value=0.0, max_value=100.0, value=float(preset_vals.get("fresh_issue_pct", 70.0)), step=5.0)
    val_ofs = col_o.number_input("OFS %", min_value=0.0, max_value=100.0, value=float(preset_vals.get("ofs_pct", 30.0)), step=5.0)

    col_rev, col_pat = st.columns(2)
    val_rev = col_rev.number_input("Revenue Growth (%)", value=float(preset_vals.get("revenue_growth_pct", 25.0)), step=5.0)
    val_pat = col_pat.number_input("Profit Growth (%)", value=float(preset_vals.get("profit_growth_pct", 20.0)), step=5.0)

    val_pe = st.number_input("P/E vs Sector Peers Ratio", min_value=0.1, max_value=10.0, value=float(preset_vals.get("pe_vs_peers", 1.0)), step=0.05, help=">1.0 indicates premium valuation; <1.0 indicates discount")

    submit_button = st.form_submit_button(" Run AI Predictions", use_container_width=True)

# Input Validation
validation_errors = []
if abs((val_fresh + val_ofs) - 100.0) > 1.0:
    validation_errors.append(f"Fresh Issue ({val_fresh}%) + OFS ({val_ofs}%) should equal 100%. (Current total: {val_fresh + val_ofs}%)")
if val_price <= 0:
    validation_errors.append("Issue Price must be strictly greater than ₹0.")

if validation_errors:
    for err in validation_errors:
        st.sidebar.error(f"⚠️ {err}")

# Build input payload
input_dict = {
    "ipo_name": val_name,
    "listing_date": val_date,
    "sector": val_sector,
    "issue_size_cr": val_size,
    "price_band_upper": val_price,
    "issue_price": val_price,
    "qib_sub": val_qib,
    "nii_sub": val_nii,
    "retail_sub": val_retail,
    "total_sub": val_total,
    "gmp_last_day": val_gmp,
    "fresh_issue_pct": val_fresh,
    "ofs_pct": val_ofs,
    "revenue_growth_pct": val_rev,
    "profit_growth_pct": val_pat,
    "pe_vs_peers": val_pe,
    "nifty_7d_trend_pct": val_nifty,
}
if val_mood_select != "Auto-Detect (HMM)":
    input_dict["market_mood"] = val_mood_select

# ---------------------------------------------------------
# Execute Inference Engine
# ---------------------------------------------------------
with st.spinner("Analyzing IPO fundamentals, subscription telemetry, HMM market regime, and peer vectors..."):
    results = predict_ipo(input_dict)

exp_gain = results["expected_gain_pct"]
pos_prob = results["positive_listing_prob"]
verdict = results["verdict"]
peers_df = results["similar_ipos"]
mood = results["market_mood"]
cluster_id = results["cluster_id"]
cluster_name = results["cluster_name"]

# ---------------------------------------------------------
# Main Page Layout: Two Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🎯 Live Prediction & Peer Intelligence", "📊 Model Comparison & Ablation Leaderboard"])

# =========================================================
# TAB 1: LIVE PREDICTION & PEER INTELLIGENCE
# =========================================================
with tab1:
    # 1. Market Mood Banner
    mood_class = f"mood-banner-{mood.lower()}"
    mood_icon = "🔥" if mood == "Hot" else ("🌤️" if mood == "Warm" else "❄️")
    mood_desc = (
        "Bullish momentum regime: historical debuts averaged high listing pops with strong retail participation."
        if mood == "Hot"
        else (
            "Neutral/moderate regime: selective debuts with sector-specific premiums."
            if mood == "Warm"
            else "Cautious/bearish regime: compressed grey-market valuations and defensive listings."
        )
    )

    st.markdown(
        f"""
        <div class="{mood_class}">
            <div>
                <span style="font-size: 1.3rem; font-weight: 700;">{mood_icon} Indian Equity Market Mood: {mood.upper()}</span>
                <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px;">{mood_desc}</div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">HMM Decoded State</span>
                <div style="font-size: 0.75rem; opacity: 0.8;">Causal Month-Prior Inference</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Output Cards
    col1, col2, col3, col4 = st.columns(4)

    # Card 1: Expected Listing Gain %
    gain_color = "#10b981" if exp_gain >= 15.0 else ("#3b82f6" if exp_gain >= 0.0 else "#ef4444")
    gain_arrow = "▲" if exp_gain >= 0.0 else "▼"
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Expected Listing Gain</div>
                <div class="metric-value-huge" style="color: {gain_color};">
                    {gain_arrow} {exp_gain:+.2f}%
                </div>
                <div class="metric-subtext">Gradient Boosting Regressor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 2: Probability of Positive Listing
    prob_pct = pos_prob * 100.0
    prob_color = "#10b981" if prob_pct >= 70.0 else ("#f59e0b" if prob_pct >= 50.0 else "#ef4444")
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Positive Listing Probability</div>
                <div class="metric-value-huge" style="color: {prob_color};">
                    {prob_pct:.1f}%
                </div>
                <div class="metric-subtext">Gradient Boosting Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 3: Investment Verdict
    badge_cls = f"badge-{verdict.lower()}"
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">SVM Investment Verdict</div>
                <div style="margin-top: 10px; margin-bottom: 8px;">
                    <span class="{badge_cls}">{verdict}</span>
                </div>
                <div class="metric-subtext">Support Vector Machine (Tuned)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 4: Cluster Archetype
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cluster Archetype</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #a78bfa; margin-top: 6px; margin-bottom: 4px;">
                    {cluster_name}
                </div>
                <div class="metric-subtext">Cluster #{cluster_id} (K-Means Out-of-Sample)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show Actual vs Predicted if testing a preset case
    if preset_vals.get("actual_gain") is not None:
        act_g = preset_vals["actual_gain"]
        act_v = preset_vals.get("actual_verdict", "N/A")
        err = exp_gain - act_g
        st.info(
            f"🎯 **Test Set Validation Benchmark for '{val_name}':** "
            f"Actual Listing Gain: **{act_g:+.2f}%** | Predicted: **{exp_gain:+.2f}%** (Error: {err:+.2f}%) | "
            f"Actual Verdict: **{act_v}** | Predicted: **{verdict}**"
        )

    st.markdown("---")

    # 3. Probability Gauge & Input Diagnostic Breakdown
    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        st.markdown("#### 🎯 Positive Debut Probability Gauge")
        # Generate custom clean Matplotlib half-gauge
        fig_gauge, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={"projection": "polar"})
        fig_gauge.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        # Angle range: 0 to pi (180 deg)
        val_norm = np.clip(pos_prob, 0.0, 1.0)
        target_angle = np.pi * (1.0 - val_norm)

        # Plot 3 background zones (Red: 0-50%, Amber: 50-75%, Green: 75-100%)
        # Polar angles go counterclockwise: pi is Left (0%), 0 is Right (100%)
        theta_red = np.linspace(np.pi, np.pi * 0.5, 50)
        theta_amber = np.linspace(np.pi * 0.5, np.pi * 0.25, 30)
        theta_green = np.linspace(np.pi * 0.25, 0.0, 30)

        ax.plot(theta_red, [1] * len(theta_red), color="#ef4444", lw=14, alpha=0.85, solid_capstyle="round")
        ax.plot(theta_amber, [1] * len(theta_amber), color="#f59e0b", lw=14, alpha=0.85)
        ax.plot(theta_green, [1] * len(theta_green), color="#10b981", lw=14, alpha=0.85, solid_capstyle="round")

        # Needle
        ax.annotate(
            "",
            xy=(target_angle, 0.95),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->, head_length=0.8, head_width=0.4", lw=3.0, color="#ffffff"),
        )
        ax.plot(0, 0, marker="o", markersize=8, color="#ffffff")

        # Gauge text
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(0, 1.15)
        ax.grid(False)
        ax.set_xticks([0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi])
        ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"], color="#94a3b8", fontsize=9, fontweight="600")
        ax.set_yticks([])

        ax.text(
            np.pi * 0.5,
            0.2,
            f"{prob_pct:.1f}%",
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=20,
            fontweight="800",
        )
        ax.text(
            np.pi * 0.5,
            -0.15,
            f"Verdict: {verdict.upper()}",
            ha="center",
            va="center",
            color=prob_color,
            fontsize=11,
            fontweight="700",
        )

        st.pyplot(fig_gauge)
        plt.close(fig_gauge)

    with col_details:
        st.markdown("#### 🔍 Core Feature Contributions")
        gmp_implied_gain = (val_gmp / val_price) * 100.0 if val_price > 0 else 0.0
        
        feat_df = pd.DataFrame([
            {"Feature Dimension": "Grey Market Implied Return", "Input Value": f"{gmp_implied_gain:+.2f}%", "Historical Impact": "Highest (+51.7% R² contribution)"},
            {"Feature Dimension": "Total Subscription Multiple", "Input Value": f"{val_total:.2f}x", "Historical Impact": "Strong positive non-linear driver"},
            {"Feature Dimension": "Institutional (QIB) Demand", "Input Value": f"{val_qib:.2f}x", "Historical Impact": "Anchor pricing validation signal"},
            {"Feature Dimension": "Nifty 50 7-Day Trend", "Input Value": f"{val_nifty:+.2f}%", "Historical Impact": "Listing week market sentiment backdrop"},
            {"Feature Dimension": "Valuation vs Peer P/E", "Input Value": f"{val_pe:.2f}x", "Historical Impact": "Discount (<1.0) aids upside headroom"},
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Top 5 Similar Historical Peer IPOs
    st.markdown("### 🤝 Top 5 Most Similar Historical Peers (k-NN Engine)")
    st.markdown(
        "Identified using multi-dimensional Euclidean distance over scaled subscription multiples, issue size, valuation, sector, and market regime. "
        "Strict anti-leakage rules enforce peer candidate listing dates prior to query date."
    )

    if not peers_df.empty:
        col_tbl, col_chart = st.columns([1.2, 0.8])

        with col_tbl:
            styled_peers = peers_df.copy()
            styled_peers["listing_gain_pct"] = styled_peers["listing_gain_pct"].apply(lambda x: f"{x:+.2f}%")
            styled_peers["total_sub"] = styled_peers["total_sub"].apply(lambda x: f"{x:.2f}x")
            styled_peers["similarity_score"] = styled_peers["similarity_score"].apply(lambda x: f"{x:.4f}")
            styled_peers.columns = [
                "Peer IPO Name", "Sector", "Total Sub", "Debut Gain %", "Listing Date", "Similarity Score"
            ]
            st.dataframe(styled_peers, use_container_width=True, hide_index=True)

        with col_chart:
            # Bar chart comparing peers' actual debut gains vs our prediction
            fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
            fig_p.patch.set_facecolor("#0e1117")
            ax_p.set_facecolor("#0e1117")

            names = [p[:16] for p in peers_df["ipo_name"]] + ["★ Predicted"]
            gains = list(peers_df["listing_gain_pct"]) + [exp_gain]
            bar_colors = ["#60a5fa" for _ in range(len(peers_df))] + [gain_color]

            bars = ax_p.barh(names, gains, color=bar_colors, edgecolor="none", height=0.6)
            ax_p.axvline(0, color="#475569", linestyle="--", lw=0.8)
            ax_p.set_xlabel("Listing Gain (%)", color="#94a3b8", fontsize=9)
            ax_p.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax_p.spines.values():
                spine.set_color("#334155")

            # Value labels on bars
            for bar in bars:
                w = bar.get_width()
                ax_p.text(
                    w + (0.5 if w >= 0 else -2.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:+.1f}%",
                    va="center",
                    color="#ffffff",
                    fontsize=8,
                    fontweight="600",
                )

            ax_p.set_title("Peer Debut Gains vs Predicted Output", color="#ffffff", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)
    else:
        st.warning("No peer IPO records retrieved.")


# =========================================================
# TAB 2: MODEL COMPARISON & ABLATION STUDY
# =========================================================
with tab2:
    st.markdown("### 🏆 Comprehensive Model Comparison Leaderboard")
    st.markdown(
        "All models were evaluated on the **identical 75-sample chronological test set** "
        "(January 2024 to December 2024) with zero data leakage. All encoders, scalers, and imputers were fitted strictly on the 225-sample training set."
    )

    if not metrics_df.empty:
        # Format the model comparison table
        display_metrics = metrics_df.copy()
        st.dataframe(
            display_metrics.style.highlight_max(subset=["R2", "Accuracy", "F1", "AUC"], color="rgba(16, 185, 129, 0.25)")
            .highlight_min(subset=["MSE", "RMSE"], color="rgba(16, 185, 129, 0.25)"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("reports/model_comparison.csv not found.")

    fig_leaderboard_path = os.path.join(PROJECT_ROOT, "reports", "figures", "18_model_comparison_leaderboard.png")
    if os.path.exists(fig_leaderboard_path):
        st.image(fig_leaderboard_path, caption="Figure 18: Model Leaderboard — Test Error (RMSE) and Classification Accuracy", use_container_width=True)

    st.markdown("---")

    # Ablation Study Section
    st.markdown("### 🔬 Ablation Study: Impact of Grey Market Premium & Macro Mood")
    st.markdown(
        "To quantify the precise value added by individual feature sets, we systematically ablated feature blocks from our best-performing Gradient Boosting architecture:"
    )

    if not ablation_df.empty:
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
    else:
        st.warning("reports/ablation_study.csv not found.")

    col_ab1, col_ab2 = st.columns(2)
    fig_ablation_path = os.path.join(PROJECT_ROOT, "reports", "figures", "19_ablation_study.png")
    fig_featimp_path = os.path.join(PROJECT_ROOT, "reports", "figures", "17_boosting_feature_importances.png")

    with col_ab1:
        if os.path.exists(fig_ablation_path):
            st.image(fig_ablation_path, caption="Figure 19: Ablation Study Metric Degradation", use_container_width=True)
    with col_ab2:
        if os.path.exists(fig_featimp_path):
            st.image(fig_featimp_path, caption="Figure 17: Gradient Boosting Top Feature Importances", use_container_width=True)

    st.markdown("---")

    # Honest Observations & Limitations Section
    st.markdown("### 📝 Honest Academic Observations & Real-World Limitations")
    st.markdown(
        """
        1. **Grey Market Premium (GMP) Dominance:**
           - The ablation study mathematically proves that removing `gmp_last_day` causes Test MSE to spike by **+51.7%** (from 85.42 to 129.59) and collapses $R^2$ to near zero (**0.003**).
           - While GMP is the single strongest price-discovery signal, in practice grey markets in India are **unregulated, opaque, and vulnerable to operator manipulation**. Artificial GMP spikes can misguide model inferences.
        
        2. **Overfitting Risks from Limited Sample Size:**
           - With 225 training records, high-capacity models (e.g. Polynomial Regression degree 4, deep decision trees) suffered catastrophic over-fitting.
           - Tree depth regularization (`max_depth=3`, conservative learning rates) and ensemble averaging were required to maintain generalization stability.
        
        3. **Regime Shifts & Macro Shocks:**
           - Sudden macroeconomic events (monetary policy rate hikes, sudden FPI capital flight, geopolitical escalations) can decouple IPO debuts from subscription demand overnight.
           - Furthermore, regulatory changes (such as SEBI's mandate compressing listing timelines from T+6 to T+3, and revised anchor lock-in phases) alter retail subscription dynamics over time.
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "IPO Listing-Gain Predictor (NSE/BSE) • Machine Learning Academic Mini-Project • Python 3.10 / Streamlit / Scikit-Learn<br>"
    "Strictly for academic demonstration. Not registered under SEBI Research Analyst regulations."
    "</div>",
    unsafe_allow_html=True,
)
