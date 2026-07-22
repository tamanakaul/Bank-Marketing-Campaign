"""
🏦 Bank Marketing Analysis Dashboard
A Streamlit + Plotly interactive dashboard for the Bank Marketing dataset.

Run with:
    streamlit run app.py

Expects a CSV with (a subset of) these columns:
    age, job, marital, education, default, balance, housing, loan,
    contact, day, month, duration, campaign, pdays, previous, poutcome,
    deposit (or 'y')  -> target column, values like yes/no

If no CSV is found/uploaded, the app auto-generates a synthetic sample
dataset so the dashboard is fully explorable out of the box.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bank Marketing Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME / CSS
# =========================================================
def inject_css(dark_mode: bool):
    if dark_mode:
        bg, card_bg, text, accent, sub = "#0e1117", "#161a23", "#f0f2f6", "#4da3ff", "#9aa4b2"
    else:
        bg, card_bg, text, accent, sub = "#f5f7fb", "#ffffff", "#1c1c1c", "#0066cc", "#5b6472"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        h1, h2, h3, h4 {{ color: {text}; }}
        section[data-testid="stSidebar"] {{ background-color: {card_bg}; }}
        div[data-testid="stMetric"] {{
            background-color: {card_bg};
            border-left: 5px solid {accent};
            border-radius: 12px;
            padding: 14px 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .kpi-title {{
            font-size: 26px;
            font-weight: 800;
            color: {text};
            margin-bottom: 0px;
        }}
        .kpi-sub {{
            font-size: 14px;
            color: {sub};
            margin-top: -6px;
        }}
        .section-card {{
            background-color: {card_bg};
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            margin-bottom: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SAMPLE DATA (fallback so the dashboard always runs)
# =========================================================
@st.cache_data
def generate_sample_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    jobs = ["admin.", "technician", "services", "management", "retired",
            "blue-collar", "unemployed", "entrepreneur", "housemaid",
            "self-employed", "student", "unknown"]
    marital = ["married", "single", "divorced"]
    education = ["primary", "secondary", "tertiary", "unknown"]
    contact = ["cellular", "telephone", "unknown"]
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcome = ["success", "failure", "other", "unknown"]
    yesno = ["yes", "no"]

    age = rng.integers(18, 90, n)
    balance = rng.normal(1400, 3000, n).round(0).astype(int)
    duration = np.abs(rng.normal(260, 200, n)).astype(int)
    campaign = rng.integers(1, 15, n)
    pdays = rng.choice([-1] + list(range(1, 400)), n)
    previous = rng.integers(0, 8, n)

    # deposit probability shaped by duration/balance to look realistic
    prob_yes = 1 / (1 + np.exp(-(duration - 300) / 150 + (balance - 1000) / 8000))
    deposit = np.where(rng.random(n) < prob_yes, "yes", "no")

    df = pd.DataFrame({
        "age": age,
        "job": rng.choice(jobs, n),
        "marital": rng.choice(marital, n, p=[0.6, 0.28, 0.12]),
        "education": rng.choice(education, n),
        "default": rng.choice(yesno, n, p=[0.02, 0.98]),
        "balance": balance,
        "housing": rng.choice(yesno, n, p=[0.55, 0.45]),
        "loan": rng.choice(yesno, n, p=[0.16, 0.84]),
        "contact": rng.choice(contact, n, p=[0.65, 0.15, 0.2]),
        "day": rng.integers(1, 29, n),
        "month": rng.choice(months, n),
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": rng.choice(poutcome, n, p=[0.08, 0.12, 0.05, 0.75]),
        "deposit": deposit,
    })
    return df


# =========================================================
# DATA LOADING & CLEANING
# =========================================================
@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize target column name
    if "deposit" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": "deposit"})

    if "deposit" in df.columns:
        df["deposit"] = df["deposit"].astype(str).str.strip().str.lower()

    for col in ["job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "poutcome"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    for col in ["age", "balance", "duration", "campaign", "pdays", "previous", "day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in ["age", "balance"] if c in df.columns])

    if "age" in df.columns:
        bins = [17, 25, 35, 45, 55, 200]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    return df


def load_data() -> pd.DataFrame:
    st.sidebar.markdown("### 📂 Data Source")
    uploaded = st.sidebar.file_uploader("Upload bank marketing CSV", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded uploaded file · {raw.shape[0]:,} rows")
        return clean_data(raw)

    # Try common local filenames
    for fname in ["bank.csv", "bank-full.csv", "bank-additional-full.csv", "bank_marketing.csv"]:
        try:
            raw = pd.read_csv(fname)
            st.sidebar.success(f"Loaded local file '{fname}' · {raw.shape[0]:,} rows")
            return clean_data(raw)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    st.sidebar.info("No CSV found — using a generated sample dataset. "
                     "Upload your own file above to replace it.")
    return clean_data(generate_sample_data())


# =========================================================
# HELPERS
# =========================================================
def has_cols(df, cols):
    return all(c in df.columns for c in cols)


def kpi_card(col, label, value, icon):
    with col:
        st.markdown(
            f"""
            <div class="section-card" style="text-align:center;">
                <div style="font-size:22px;">{icon}</div>
                <div class="kpi-title">{value}</div>
                <div class="kpi-sub">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# LOAD DATA + SIDEBAR
# =========================================================
st.sidebar.title("🏦 Bank Marketing")
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)
inject_css(dark_mode)

df = load_data()

st.sidebar.markdown("### 🔍 Filters")

filtered = df.copy()

if "age" in df.columns:
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
    filtered = filtered[filtered["age"].between(*age_range)]

def multiselect_filter(col_name, label):
    global filtered
    if col_name in df.columns:
        options = sorted(df[col_name].dropna().unique().tolist())
        chosen = st.sidebar.multiselect(label, options, default=options)
        filtered = filtered[filtered[col_name].isin(chosen)]

multiselect_filter("job", "Job")
multiselect_filter("marital", "Marital Status")
multiselect_filter("education", "Education")
multiselect_filter("month", "Month")
multiselect_filter("contact", "Contact Type")
multiselect_filter("housing", "Housing Loan")
multiselect_filter("loan", "Personal Loan")
multiselect_filter("deposit", "Deposit (Yes/No)")

st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download Filtered Data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_bank_data.csv",
    mime="text/csv",
    use_container_width=True,
)

PAGES = ["Home", "Customer Analysis", "Financial Analysis",
         "Campaign Analysis", "Deposit Prediction Insights", "Data Explorer"]
PAGE_ICONS = ["house", "people-fill", "cash-coin", "telephone-fill", "bullseye", "table"]
PAGE_EMOJI = {
    "Home": "🏠 Home",
    "Customer Analysis": "📊 Customer Analysis",
    "Financial Analysis": "💰 Financial Analysis",
    "Campaign Analysis": "📞 Campaign Analysis",
    "Deposit Prediction Insights": "🎯 Deposit Prediction Insights",
    "Data Explorer": "📋 Data Explorer",
}

st.markdown(
    """
    <div style="text-align:center; padding: 4px 0 10px 0;">
        <span style="font-size:30px; font-weight:800;">🏦 Bank Marketing Analysis Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if HAS_OPTION_MENU:
    nav_choice = option_menu(
        menu_title=None,
        options=PAGES,
        icons=PAGE_ICONS,
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "6px 4px", "background-color": "transparent"},
            "icon": {"font-size": "15px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px 2px", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#0066cc"},
        },
    )
    page = PAGE_EMOJI[nav_choice]
else:
    nav_choice = st.radio(
        "Navigate",
        PAGES,
        horizontal=True,
        label_visibility="collapsed",
    )
    page = PAGE_EMOJI[nav_choice]

st.markdown("---")

if filtered.empty:
    st.warning("No records match the current filters. Please widen your filter selection.")
    st.stop()


# =========================================================
# KPI ROW (shown on every page except Data Explorer for space)
# =========================================================
def render_kpis(d):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "Total Customers", f"{len(d):,}", "👥")
    kpi_card(c2, "Avg Balance", f"${d['balance'].mean():,.0f}" if "balance" in d else "—", "💰")

    if "deposit" in d.columns:
        accepted = (d["deposit"] == "yes").sum()
        rejected = (d["deposit"] == "no").sum()
        conv = accepted / len(d) * 100 if len(d) else 0
    else:
        accepted = rejected = conv = 0

    kpi_card(c3, "Deposit Accepted", f"{accepted:,}", "✅")
    kpi_card(c4, "Deposit Rejected", f"{rejected:,}", "❌")
    kpi_card(c5, "Conversion Rate", f"{conv:.1f}%", "📈")
    kpi_card(c6, "Avg Call Duration",
              f"{d['duration'].mean():,.0f}s" if "duration" in d else "—", "📞")


# =========================================================
# PAGE: HOME
# =========================================================
if page == "🏠 Home":
    st.caption("A professional overview of customer demographics, finances, and campaign performance.")
    render_kpis(filtered)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Deposit Distribution")
        if "deposit" in filtered.columns:
            counts = filtered["deposit"].value_counts().reset_index()
            counts.columns = ["deposit", "count"]
            fig = px.pie(counts, names="deposit", values="count",
                         color="deposit",
                         color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"},
                         hole=0.0)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Campaign Analysis")
        if has_cols(filtered, ["month", "deposit"]):
            month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
            monthly = (filtered.groupby(["month", "deposit"]).size()
                       .reset_index(name="count"))
            monthly["month"] = pd.Categorical(monthly["month"], categories=month_order, ordered=True)
            monthly = monthly.sort_values("month")
            fig = px.line(monthly, x="month", y="count", color="deposit", markers=True,
                          color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Quick Highlights")
    c1, c2, c3 = st.columns(3)
    if "job" in filtered.columns:
        top_job = filtered["job"].value_counts().idxmax()
        c1.info(f"**Most common job:** {top_job.title()}")
    if has_cols(filtered, ["month", "deposit"]):
        best_month = (filtered[filtered["deposit"] == "yes"]["month"]
                      .value_counts().idxmax()) if (filtered["deposit"] == "yes").any() else "—"
        c2.info(f"**Best month for deposits:** {best_month.title()}")
    if "age" in filtered.columns:
        c3.info(f"**Average customer age:** {filtered['age'].mean():.1f} yrs")


# =========================================================
# PAGE: CUSTOMER ANALYSIS
# =========================================================
elif page == "📊 Customer Analysis":
    st.subheader("📊 Customer Analysis")
    render_kpis(filtered)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customers by Job")
        if "job" in filtered.columns:
            job_counts = filtered["job"].value_counts().reset_index()
            job_counts.columns = ["job", "count"]
            fig = px.bar(job_counts.sort_values("count"), x="count", y="job",
                        orientation="h", color="count", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Education Distribution")
        if "education" in filtered.columns:
            edu_counts = filtered["education"].value_counts().reset_index()
            edu_counts.columns = ["education", "count"]
            fig = px.pie(edu_counts, names="education", values="count", hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Marital Status Distribution")
        if "marital" in filtered.columns:
            mar_counts = filtered["marital"].value_counts().reset_index()
            mar_counts.columns = ["marital", "count"]
            fig = px.pie(mar_counts, names="marital", values="count")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Age Distribution")
        if "age" in filtered.columns:
            fig = px.histogram(filtered, x="age", nbins=30, color_discrete_sequence=["#0066cc"])
            fig.update_layout(bargap = 0.1)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Deposit by Age Group")
    if has_cols(filtered, ["age_group", "deposit"]):
        age_dep = (filtered.groupby(["age_group", "deposit"], observed=True).size()
                   .reset_index(name="count"))
        fig = px.bar(age_dep, x="age_group", y="count", color="deposit",
                    barmode="group",
                    color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Education → Job → Deposit (Sunburst)")
    if has_cols(filtered, ["education", "job", "deposit"]):
        keep_jobs = ["technician", "management", "housemaid", "entrepreneur", "admin."]
        sb_src = filtered.copy()
        sb_src["job_grouped"] = np.where(sb_src["job"].isin(keep_jobs), sb_src["job"], "others")
        sb = sb_src.groupby(["education", "job_grouped", "deposit"]).size().reset_index(name="count")
        fig = px.sunburst(sb, path=["education", "job_grouped", "deposit"], values="count",
                          color="deposit", color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c", "(?)": "#ccc"})
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: FINANCIAL ANALYSIS
# =========================================================
elif page == "💰 Financial Analysis":
    st.subheader("💰 Financial Analysis")
    render_kpis(filtered)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Balance by Job")
        if has_cols(filtered, ["job", "balance"]):
            avg_bal = filtered.groupby("job")["balance"].mean().reset_index().sort_values("balance")
            fig = px.bar(avg_bal, x="job", y="balance", color="balance", color_continuous_scale="Teal")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Balance by Education")
        if has_cols(filtered, ["education", "balance"]):
            avg_bal = filtered.groupby("education")["balance"].mean().reset_index().sort_values("balance")
            fig = px.bar(avg_bal, x="education", y="balance", color="balance", color_continuous_scale="Purples")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Balance Distribution")
        if "balance" in filtered.columns:
            fig = px.histogram(filtered, x="balance", nbins=40, color_discrete_sequence=["#27ae60"])
            fig.update_layout(bargap = 0.1)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Correlation Heatmap")
        num_cols = [c for c in ["age", "balance", "duration", "campaign", "previous", "pdays"]
                    if c in filtered.columns]
        if len(num_cols) >= 2:
            corr = filtered[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Balance (colored by Deposit)")
    if has_cols(filtered, ["age", "balance", "deposit"]):
        fig = px.scatter(filtered, x="age", y="balance", color="deposit",
                         opacity=0.6, color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Call Duration by Deposit (Box Plot)")
        if has_cols(filtered, ["duration", "deposit"]):
            fig = px.box(filtered, x="deposit", y="duration", color="deposit",
                        color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Balance by Deposit (Violin Plot)")
        if has_cols(filtered, ["balance", "deposit"]):
            fig = px.violin(filtered, x="deposit", y="balance", color="deposit", box=True,
                            color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: CAMPAIGN ANALYSIS
# =========================================================
elif page == "📞 Campaign Analysis":
    st.subheader("📞 Campaign Analysis")
    render_kpis(filtered)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Contact Method")
        if "contact" in filtered.columns:
            cc = filtered["contact"].value_counts().reset_index()
            cc.columns = ["contact", "count"]
            fig = px.pie(cc, names="contact", values="count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Previous Campaign Outcome")
        if "poutcome" in filtered.columns:
            pc = filtered["poutcome"].value_counts().reset_index()
            pc.columns = ["poutcome", "count"]
            fig = px.pie(pc, names="poutcome", values="count")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Housing Loan vs Deposit")
        if has_cols(filtered, ["housing", "deposit"]):
            hd = filtered.groupby(["housing", "deposit"]).size().reset_index(name="count")
            fig = px.bar(hd, x="housing", y="count", color="deposit", barmode="group",
                        color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Personal Loan vs Deposit")
        if has_cols(filtered, ["loan", "deposit"]):
            ld = filtered.groupby(["loan", "deposit"]).size().reset_index(name="count")
            fig = px.bar(ld, x="loan", y="count", color="deposit", barmode="group",
                        color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Default Credit vs Deposit")
        if has_cols(filtered, ["default", "deposit"]):
            dd = filtered.groupby(["default", "deposit"]).size().reset_index(name="count")
            fig = px.bar(dd, x="default", y="count", color="deposit", barmode="group",
                        color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Campaign Calls Distribution")
        if "campaign" in filtered.columns:
            fig = px.histogram(filtered, x="campaign", nbins=20, color_discrete_sequence=["#e67e22"])
            fig.update_layout(bargap = 0.1)
            st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Previous Contact Count")
        if "previous" in filtered.columns:
            fig = px.bar(filtered["previous"].value_counts().sort_index().reset_index(
                        name="count").rename(columns={"index": "previous"}),
                        x="previous", y="count", color_discrete_sequence=["#8e44ad"])
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        st.subheader("Call Duration Distribution")
        if "duration" in filtered.columns:
            fig = px.histogram(filtered, x="duration", nbins=40, color_discrete_sequence=["#16a085"])
            fig.update_layout(bargap = 0.1)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Job → Deposit (Treemap)")
    if has_cols(filtered, ["job", "deposit"]):
        tm = filtered.groupby(["job", "deposit"]).size().reset_index(name="count")
        fig = px.treemap(tm, path=["job", "deposit"], values="count", color="deposit",
                         color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c", "(?)": "#ccc"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Campaign Funnel: Contacts → Reached → Converted")
    if has_cols(filtered, ["contact", "deposit"]):
        total = len(filtered)
        reached = filtered[filtered["contact"] != "unknown"].shape[0] if "unknown" in filtered["contact"].unique() else total
        converted = (filtered["deposit"] == "yes").sum()
        fig = go.Figure(go.Funnel(
            y=["Total Campaign Contacts", "Successfully Reached", "Deposit Converted"],
            x=[total, reached, converted],
            textinfo="value+percent initial",
            marker={"color": ["#3498db", "#f39c12", "#2ecc71"]},
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Duration (Bubble = Balance, Color = Deposit)")
    if has_cols(filtered, ["age", "duration", "balance", "deposit"]):
        bub = filtered.copy()
        bub["bubble_size"] = bub["balance"].clip(lower=0) + 1
        fig = px.scatter(bub, x="age", y="duration", size="bubble_size", color="deposit",
                         opacity=0.6, size_max=40,
                         color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: DEPOSIT PREDICTION INSIGHTS
# =========================================================
elif page == "🎯 Deposit Prediction Insights":
    st.subheader("🎯 Deposit Prediction Insights")
    render_kpis(filtered)
    st.markdown("---")

    num_cols = [c for c in ["age", "balance", "duration", "campaign", "previous"]
                if c in filtered.columns]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Profile Radar: Yes vs No")
        if num_cols and "deposit" in filtered.columns:
            grouped = filtered.groupby("deposit")[num_cols].mean()
            norm = (grouped - filtered[num_cols].min()) / (filtered[num_cols].max() - filtered[num_cols].min() + 1e-9)
            fig = go.Figure()
            colors = {"yes": "#2ecc71", "no": "#e74c3c"}
            for dep in norm.index:
                fig.add_trace(go.Scatterpolar(
                    r=norm.loc[dep].values.tolist() + [norm.loc[dep].values[0]],
                    theta=num_cols + [num_cols[0]],
                    fill="toself",
                    name=dep,
                    line_color=colors.get(dep, "#888"),
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation with Deposit (numeric proxy)")
        if num_cols and "deposit" in filtered.columns:
            tmp = filtered.copy()
            tmp["deposit_flag"] = (tmp["deposit"] == "yes").astype(int)
            corrs = tmp[num_cols + ["deposit_flag"]].corr()["deposit_flag"].drop("deposit_flag")
            corrs = corrs.reset_index()
            corrs.columns = ["feature", "correlation"]
            fig = px.bar(corrs.sort_values("correlation"), x="correlation", y="feature",
                        orientation="h", color="correlation", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Duration is the strongest signal — distribution by outcome")
    if has_cols(filtered, ["duration", "deposit"]):
        fig = px.histogram(filtered, x="duration", color="deposit", barmode="overlay",
                           opacity=0.6, nbins=40,
                           color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Conversion Rate by Job")
    if has_cols(filtered, ["job", "deposit"]):
        conv = filtered.groupby("job")["deposit"].apply(lambda s: (s == "yes").mean() * 100).reset_index()
        conv.columns = ["job", "conversion_rate"]
        fig = px.bar(conv.sort_values("conversion_rate"), x="conversion_rate", y="job",
                    orientation="h", color="conversion_rate", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: DATA EXPLORER
# =========================================================
elif page == "📋 Data Explorer":
    st.subheader("📋 Data Explorer")
    st.caption(f"{len(filtered):,} rows match the current sidebar filters.")

    search = st.text_input("🔍 Global search (matches any column, case-insensitive)")
    explorer_df = filtered.copy()

    if search:
        mask = explorer_df.astype(str).apply(
            lambda row: row.str.contains(search, case=False, na=False)
        ).any(axis=1)
        explorer_df = explorer_df[mask]

    sort_col = st.selectbox("Sort by", explorer_df.columns.tolist(), index=0)
    sort_asc = st.checkbox("Ascending", value=True)
    explorer_df = explorer_df.sort_values(sort_col, ascending=sort_asc)

    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    total_pages = max(1, (len(explorer_df) - 1) // page_size + 1)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page_num - 1) * page_size
    end = start + page_size

    st.dataframe(explorer_df.iloc[start:end], use_container_width=True, height=460)
    st.caption(f"Showing rows {start + 1}–{min(end, len(explorer_df))} of {len(explorer_df):,} (page {page_num} of {total_pages})")

    st.download_button(
        "📥 Download Search Results (CSV)",
        data=explorer_df.to_csv(index=False).encode("utf-8"),
        file_name="bank_data_search_results.csv",
        mime="text/csv",
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Built with Streamlit + Plotly · Bank Marketing Analysis Dashboard")