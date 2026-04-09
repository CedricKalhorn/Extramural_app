import os
from datetime import date

import pandas as pd
import streamlit as st


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Duco Monitoring App", page_icon="📊", layout="wide")

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "duco_daily_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

EXPECTED_COLUMNS = {
    "date": "",
    "blokje_groen": 0,
    "blokje_oranje": 0,
    "blokje_rood": 0,
    "cat_score": 7,
    "cat_flag": "Normaal",
    "ouder_afwijkend_gedrag": 0,   # 1x/week
    "leerkracht_score": 7,         # 1x/week
    "wearable_steps_change_pct": 0.0,
    "wearable_hrv_change_pct": 0.0,
    "wearable_sleep_change_pct": 0.0,
    "school_vermoeidheid": False,
    "kinderpsycholoog_signaal": False,
    "notities": "",
}


# =========================
# DATA HELPERS
# =========================
def init_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EXPECTED_COLUMNS.keys()))


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, default in EXPECTED_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    df = df[list(EXPECTED_COLUMNS.keys())]

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        numeric_int_cols = [
            "blokje_groen",
            "blokje_oranje",
            "blokje_rood",
            "cat_score",
            "ouder_afwijkend_gedrag",
            "leerkracht_score",
        ]
        for col in numeric_int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        numeric_float_cols = [
            "wearable_steps_change_pct",
            "wearable_hrv_change_pct",
            "wearable_sleep_change_pct",
        ]
        for col in numeric_float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["school_vermoeidheid"] = (
            df["school_vermoeidheid"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
        df["kinderpsycholoog_signaal"] = (
            df["kinderpsycholoog_signaal"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

        df["cat_flag"] = df["cat_flag"].fillna("Normaal")
        df["notities"] = df["notities"].fillna("")

    return df


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        return init_dataframe()

    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        return init_dataframe()

    if df.empty and len(df.columns) == 0:
        return init_dataframe()

    return ensure_schema(df)


def save_data(df: pd.DataFrame) -> None:
    df_to_save = ensure_schema(df)
    if not df_to_save.empty:
        df_to_save["date"] = pd.to_datetime(df_to_save["date"], errors="coerce").astype(str)
    df_to_save.to_csv(DATA_FILE, index=False)


def upsert_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = ensure_schema(df)
    current_date = row["date"]

    if not df.empty and current_date in set(df["date"]):
        df = df[df["date"] != current_date]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = ensure_schema(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================
# FEATURE ENGINEERING
# =========================
def add_blokje_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["blokje_totaal"] = (
        df["blokje_groen"] +
        df["blokje_oranje"] +
        df["blokje_rood"]
    )

    df["blokje_totaal"] = df["blokje_totaal"].replace(0, pd.NA)

    df["pct_groen"] = (df["blokje_groen"] / df["blokje_totaal"]) * 100
    df["pct_oranje"] = (df["blokje_oranje"] / df["blokje_totaal"]) * 100
    df["pct_rood"] = (df["blokje_rood"] / df["blokje_totaal"]) * 100

    df["pct_groen"] = df["pct_groen"].fillna(0)
    df["pct_oranje"] = df["pct_oranje"].fillna(0)
    df["pct_rood"] = df["pct_rood"].fillna(0)

    return df


def cat_flag_to_numeric(value: str) -> int:
    mapping = {"Normaal": 0, "Borderline": 1, "Abnormaal": 2}
    return mapping.get(value, 0)


def bool_to_int(value) -> int:
    return 1 if bool(value) else 0


def get_recent_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    max_date = max(df["date"])
    min_date = pd.to_datetime(max_date) - pd.Timedelta(days=days - 1)
    mask = pd.to_datetime(df["date"]) >= min_date
    return df.loc[mask].copy()


def safe_mean(series: pd.Series):
    if series.empty:
        return None
    return float(series.mean())


def add_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    iso = dt.dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)
    return df


def get_weekly_observation_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zet ouder- en leerkrachtmetingen om naar weekniveau.
    Bij meerdere entries in dezelfde week wordt de laatste entry van die week gebruikt.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "week", "year_week", "ouder_afwijkend_gedrag", "leerkracht_score", "date"])

    weekly_df = add_week_columns(df)
    weekly_df = weekly_df.sort_values("date")
    weekly_df = weekly_df.groupby(["year", "week", "year_week"], as_index=False).last()

    return weekly_df[["year", "week", "year_week", "date", "ouder_afwijkend_gedrag", "leerkracht_score"]]


# =========================
# METRICS
# =========================
def compute_metrics(df: pd.DataFrame) -> dict:
    df = ensure_schema(df)

    if df.empty:
        return {}

    df = add_blokje_features(df)

    df["cat_flag_num"] = df["cat_flag"].apply(cat_flag_to_numeric)
    df["psych_signaal_num"] = df["kinderpsycholoog_signaal"].apply(bool_to_int)
    df["school_vermoeidheid_num"] = df["school_vermoeidheid"].apply(bool_to_int)

    last_7 = get_recent_window(df, 7)
    last_14 = get_recent_window(df, 14)
    last_28 = get_recent_window(df, 28)

    weekly_obs = get_weekly_observation_df(df)
    weekly_obs = weekly_obs.sort_values("date")

    last_1w_obs = weekly_obs.tail(1)
    last_2w_obs = weekly_obs.tail(2)
    last_4w_obs = weekly_obs.tail(4)

    metrics = {
        "entries_total": len(df),

        # Stoplicht
        "pct_rood_7_mean": safe_mean(last_7["pct_rood"]) if not last_7.empty else None,
        "pct_rood_14_mean": safe_mean(last_14["pct_rood"]) if not last_14.empty else None,
        "pct_rood_28_mean": safe_mean(last_28["pct_rood"]) if not last_28.empty else None,

        "pct_groen_7_mean": safe_mean(last_7["pct_groen"]) if not last_7.empty else None,
        "pct_oranje_7_mean": safe_mean(last_7["pct_oranje"]) if not last_7.empty else None,

        "rood_count_7_sum": int(last_7["blokje_rood"].sum()) if not last_7.empty else 0,
        "oranje_count_7_sum": int(last_7["blokje_oranje"].sum()) if not last_7.empty else 0,
        "groen_count_7_sum": int(last_7["blokje_groen"].sum()) if not last_7.empty else 0,

        # Vragenlijst
        "cat_score_7_mean": safe_mean(last_7["cat_score"]) if not last_7.empty else None,
        "cat_score_14_mean": safe_mean(last_14["cat_score"]) if not last_14.empty else None,
        "cat_score_28_mean": safe_mean(last_28["cat_score"]) if not last_28.empty else None,

        "cat_flag_7_abnormal_count": int((last_7["cat_flag"] == "Abnormaal").sum()) if not last_7.empty else 0,
        "cat_flag_14_abnormal_count": int((last_14["cat_flag"] == "Abnormaal").sum()) if not last_14.empty else 0,
        "cat_flag_7_borderline_or_abnormal_count": int(
            last_7["cat_flag"].isin(["Borderline", "Abnormaal"]).sum()
        ) if not last_7.empty else 0,

        # Ouders 1x/week
        "ouder_1w": int(last_1w_obs["ouder_afwijkend_gedrag"].sum()) if not last_1w_obs.empty else 0,
        "ouder_2w_sum": int(last_2w_obs["ouder_afwijkend_gedrag"].sum()) if not last_2w_obs.empty else 0,
        "ouder_4w_sum": int(last_4w_obs["ouder_afwijkend_gedrag"].sum()) if not last_4w_obs.empty else 0,

        # Leerkracht 1x/week
        "leerkracht_1w": safe_mean(last_1w_obs["leerkracht_score"]) if not last_1w_obs.empty else None,
        "leerkracht_4w_mean": safe_mean(last_4w_obs["leerkracht_score"]) if not last_4w_obs.empty else None,

        # Wearable
        "steps_7_mean": safe_mean(last_7["wearable_steps_change_pct"]) if not last_7.empty else None,
        "steps_14_mean": safe_mean(last_14["wearable_steps_change_pct"]) if not last_14.empty else None,
        "steps_28_mean": safe_mean(last_28["wearable_steps_change_pct"]) if not last_28.empty else None,

        "hrv_7_mean": safe_mean(last_7["wearable_hrv_change_pct"]) if not last_7.empty else None,
        "hrv_14_mean": safe_mean(last_14["wearable_hrv_change_pct"]) if not last_14.empty else None,
        "hrv_28_mean": safe_mean(last_28["wearable_hrv_change_pct"]) if not last_28.empty else None,

        "sleep_7_mean": safe_mean(last_7["wearable_sleep_change_pct"]) if not last_7.empty else None,
        "sleep_14_mean": safe_mean(last_14["wearable_sleep_change_pct"]) if not last_14.empty else None,
        "sleep_28_mean": safe_mean(last_28["wearable_sleep_change_pct"]) if not last_28.empty else None,

        # Overig
        "school_vermoeidheid_28_count": int(last_28["school_vermoeidheid_num"].sum()) if not last_28.empty else 0,
        "psych_signaal_14_count": int(last_14["psych_signaal_num"].sum()) if not last_14.empty else 0,
        "psych_signaal_28_count": int(last_28["psych_signaal_num"].sum()) if not last_28.empty else 0,
    }

    return metrics


# =========================
# ALERTS
# =========================
def detect_negative_trends(metrics: dict) -> list[str]:
    alerts = []

    if not metrics:
        return alerts

    if metrics.get("pct_rood_7_mean") is not None and metrics["pct_rood_7_mean"] >= 20:
        alerts.append("Negatieve trend gedetecteerd in stoplichtblokje: percentage rood is verhoogd.")

    if metrics.get("pct_groen_7_mean") is not None and metrics["pct_groen_7_mean"] <= 40:
        alerts.append("Negatieve trend gedetecteerd in stoplichtblokje: percentage groen is laag.")

    if metrics.get("cat_flag_7_abnormal_count", 0) >= 3:
        alerts.append("Negatieve trend gedetecteerd in emotionele vragenlijst: meerdere abnormale scores in 7 dagen.")

    if metrics.get("cat_score_7_mean") is not None and metrics["cat_score_7_mean"] <= 5:
        alerts.append("Negatieve trend gedetecteerd in emotionele vragenlijstscore.")

    if metrics.get("ouder_1w", 0) >= 5:
        alerts.append("Negatieve trend gedetecteerd in ouderobservaties.")

    if metrics.get("leerkracht_4w_mean") is not None and metrics["leerkracht_4w_mean"] < 5:
        alerts.append("Negatieve trend gedetecteerd in leerkrachtobservaties.")

    if metrics.get("steps_7_mean") is not None and metrics["steps_7_mean"] <= -20:
        alerts.append("Negatieve trend gedetecteerd in wearable activiteit.")

    if metrics.get("hrv_7_mean") is not None and metrics["hrv_7_mean"] <= -10:
        alerts.append("Negatieve trend gedetecteerd in wearable HRV.")

    if metrics.get("sleep_7_mean") is not None and metrics["sleep_7_mean"] <= -15:
        alerts.append("Negatieve trend gedetecteerd in wearable slaap.")

    if metrics.get("school_vermoeidheid_28_count", 0) >= 5:
        alerts.append("Negatieve trend gedetecteerd in schoolvermoeidheid.")

    if metrics.get("psych_signaal_14_count", 0) >= 1:
        alerts.append("Negatieve trend gedetecteerd vanuit kinderpsycholoog.")

    return alerts


# =========================
# INTERVENTIES
# =========================
def intervention_rules(metrics: dict) -> dict:
    if not metrics:
        return {}

    results = {}

    add_on_checks = {
        "Stoplicht: week lang >5% meer rood / duidelijke rode trend": (metrics.get("pct_rood_7_mean") or 0) >= 5,
        "Wearable stappen gedaald 20-30%": (
            metrics.get("steps_7_mean") is not None and metrics["steps_7_mean"] <= -20
        ),
        "Wearable HRV gedaald ≥20%": (
            metrics.get("hrv_7_mean") is not None and metrics["hrv_7_mean"] <= -20
        ),
        "Slaap gedaald ≥15%": (
            metrics.get("sleep_7_mean") is not None and metrics["sleep_7_mean"] <= -15
        ),
        "Vragenlijst borderline/abnormaal": metrics.get("cat_flag_7_borderline_or_abnormal_count", 0) >= 1,
    }
    results["Add-on programma"] = add_on_checks

    hulpkracht_checks = {
        "Stoplicht 20-30% meer rood gedurende maand": (metrics.get("pct_rood_28_mean") or 0) >= 20,
        "Wearable activiteit tijdens periode ≥20% lager": (
            metrics.get("steps_28_mean") is not None and metrics["steps_28_mean"] <= -20
        ),
        "Leerkrachtobservatie gemiddeld <5 over 4 weken": (
            metrics.get("leerkracht_4w_mean") is not None and metrics["leerkracht_4w_mean"] < 5
        ),
        "Kinderpsycholoog signaal aanwezig": metrics.get("psych_signaal_28_count", 0) >= 1,
    }
    results["Hulpkracht in de klas"] = hulpkracht_checks

    muziek_checks = {
        "Stoplicht ≥20% rood gedurende 2 weken": (metrics.get("pct_rood_14_mean") or 0) >= 20,
        "HRV chronisch laag (≥20% daling)": (
            metrics.get("hrv_14_mean") is not None and metrics["hrv_14_mean"] <= -20
        ),
        "Slaap verslechtering ≥15%": (
            metrics.get("sleep_14_mean") is not None and metrics["sleep_14_mean"] <= -15
        ),
        "3 abnormale vragenlijstscores in week": metrics.get("cat_flag_7_abnormal_count", 0) >= 3,
        "10 afwijkende gedragingen ouders in 1 week": metrics.get("ouder_1w", 0) >= 10,
    }
    results["Muziektherapie"] = muziek_checks

    game_checks = {
        "Stoplicht ≥5% rood op sociaal/emotioneel gedrag": (metrics.get("pct_rood_7_mean") or 0) >= 5,
        "HRV verslechterd ≥10%": (
            metrics.get("hrv_7_mean") is not None and metrics["hrv_7_mean"] <= -10
        ),
        "Slaap gedaald ≥15%": (
            metrics.get("sleep_7_mean") is not None and metrics["sleep_7_mean"] <= -15
        ),
        "5 afwijkende gedragingen ouders in 1 week": metrics.get("ouder_1w", 0) >= 5,
    }
    results["Online game community"] = game_checks

    cgt_checks = {
        "Stoplicht ≥20% rood gedurende ≥2 weken": (metrics.get("pct_rood_14_mean") or 0) >= 20,
        "Stappen 20-30% lager over 14 dagen": (
            metrics.get("steps_14_mean") is not None and metrics["steps_14_mean"] <= -20
        ),
        "HRV ≥20% lager over 14 dagen": (
            metrics.get("hrv_14_mean") is not None and metrics["hrv_14_mean"] <= -20
        ),
        "Slaap ≥20% slechter over 14 dagen": (
            metrics.get("sleep_14_mean") is not None and metrics["sleep_14_mean"] <= -20
        ),
        "3 abnormale vragenlijstscores in 14 dagen": metrics.get("cat_flag_14_abnormal_count", 0) >= 3,
        "20 afwijkende gedragingen ouders in 2 weken": metrics.get("ouder_2w_sum", 0) >= 20,
        "Kinderpsycholoog signaal aanwezig": metrics.get("psych_signaal_14_count", 0) >= 1,
    }
    results["CGT"] = cgt_checks

    return results


def summarize_interventions(results: dict) -> dict:
    summary = {}
    for intervention, checks in results.items():
        total = len(checks)
        passed = sum(bool(v) for v in checks.values())
        summary[intervention] = {
            "passed": passed,
            "total": total,
            "advice": passed >= max(1, total // 2)
        }
    return summary


def show_metric_card(title: str, value, suffix: str = ""):
    if value is None:
        display_value = "-"
    elif isinstance(value, float):
        display_value = f"{value:.1f}{suffix}"
    else:
        display_value = f"{value}{suffix}"
    st.metric(title, display_value)


# =========================
# LOAD
# =========================
df = load_data()


# =========================
# SESSION ALERT MEMORY
# =========================
if "shown_alerts" not in st.session_state:
    st.session_state["shown_alerts"] = set()


# =========================
# HEADER
# =========================
st.title("Duco Monitoring App")
st.caption("Dagelijkse registratie, trendanalyse en interventie-advies")


# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Invoer")

input_date = st.sidebar.date_input("Datum", value=date.today())

st.sidebar.subheader("Stoplichtblokje sensor-data")
blokje_groen = st.sidebar.number_input("Aantal keer groen vandaag", min_value=0, max_value=500, value=0, step=1)
blokje_oranje = st.sidebar.number_input("Aantal keer oranje vandaag", min_value=0, max_value=500, value=0, step=1)
blokje_rood = st.sidebar.number_input("Aantal keer rood vandaag", min_value=0, max_value=500, value=0, step=1)

cat_score = st.sidebar.slider("CAT emotionele vragenlijst score", min_value=1, max_value=10, value=7)
cat_flag = st.sidebar.selectbox("CAT classificatie", ["Normaal", "Borderline", "Abnormaal"])

st.sidebar.subheader("Ouderobservatie (1x/week)")
ouder_afwijkend_gedrag = st.sidebar.number_input(
    "Aantal afwijkende gedragingen deze week",
    min_value=0,
    max_value=50,
    value=0,
    step=1
)

st.sidebar.subheader("Leerkrachtobservatie (1x/week)")
leerkracht_score = st.sidebar.slider(
    "Leerkracht observatiescore deze week",
    min_value=1,
    max_value=10,
    value=7
)

st.sidebar.subheader("Wearable")
wearable_steps_change_pct = st.sidebar.number_input(
    "Wearable stappen verandering t.o.v. baseline (%)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=1.0
)

wearable_hrv_change_pct = st.sidebar.number_input(
    "Wearable HRV verandering t.o.v. baseline (%)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=1.0
)

wearable_sleep_change_pct = st.sidebar.number_input(
    "Wearable slaap verandering t.o.v. baseline (%)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=1.0
)

school_vermoeidheid = st.sidebar.checkbox("Schoolvermoeidheid / middagdip aanwezig")
kinderpsycholoog_signaal = st.sidebar.checkbox("Signaal van kinderpsycholoog")
notities = st.sidebar.text_area("Notities")

if st.sidebar.button("Opslaan"):
    row = {
        "date": input_date,
        "blokje_groen": blokje_groen,
        "blokje_oranje": blokje_oranje,
        "blokje_rood": blokje_rood,
        "cat_score": cat_score,
        "cat_flag": cat_flag,
        "ouder_afwijkend_gedrag": ouder_afwijkend_gedrag,
        "leerkracht_score": leerkracht_score,
        "wearable_steps_change_pct": wearable_steps_change_pct,
        "wearable_hrv_change_pct": wearable_hrv_change_pct,
        "wearable_sleep_change_pct": wearable_sleep_change_pct,
        "school_vermoeidheid": school_vermoeidheid,
        "kinderpsycholoog_signaal": kinderpsycholoog_signaal,
        "notities": notities,
    }

    df = upsert_row(df, row)
    save_data(df)
    st.sidebar.success("Registratie opgeslagen.")


# =========================
# MAIN CONTENT
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Trendanalyse",
    "Interventies",
    "Data"
])

with tab1:
    st.subheader("Overzicht")

    if df.empty:
        st.info("Nog geen data opgeslagen.")
    else:
        metrics = compute_metrics(df)
        alerts = detect_negative_trends(metrics)

        for alert in alerts:
            if alert not in st.session_state["shown_alerts"]:
                st.toast(alert, icon="⚠️")
                st.session_state["shown_alerts"].add(alert)

        if alerts:
            st.error("Waarschuwingen")
            for a in alerts:
                st.write(f"- {a}")
        else:
            st.success("Geen negatieve trends gedetecteerd.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_metric_card("Aantal entries", metrics.get("entries_total"))
            show_metric_card("Rood 7d gemiddelde", metrics.get("pct_rood_7_mean"), "%")
        with c2:
            show_metric_card("CAT score 7d", metrics.get("cat_score_7_mean"))
            show_metric_card("Ouder signalen 1w", metrics.get("ouder_1w"))
        with c3:
            show_metric_card("Leerkracht 4w", metrics.get("leerkracht_4w_mean"))
            show_metric_card("Stappen 7d", metrics.get("steps_7_mean"), "%")
        with c4:
            show_metric_card("HRV 7d", metrics.get("hrv_7_mean"), "%")
            show_metric_card("Slaap 7d", metrics.get("sleep_7_mean"), "%")

        st.markdown("### Laatste registraties")
        view_df = df.sort_values("date", ascending=False).head(10).copy()
        st.dataframe(view_df, use_container_width=True)

with tab2:
    st.subheader("Trendanalyse")

    if df.empty:
        st.info("Nog geen data beschikbaar.")
    else:
        chart_df = df.sort_values("date").copy()
        chart_df = add_blokje_features(chart_df)

        st.markdown("**Stoplichtblokje aantallen per dag**")
        st.bar_chart(chart_df.set_index("date")[["blokje_groen", "blokje_oranje", "blokje_rood"]])

        st.markdown("**Stoplichtblokje percentages per dag**")
        st.line_chart(chart_df.set_index("date")[["pct_groen", "pct_oranje", "pct_rood"]])

        st.markdown("**CAT score**")
        st.line_chart(chart_df.set_index("date")[["cat_score"]])

        st.markdown("**Wearable trends**")
        st.line_chart(
            chart_df.set_index("date")[[
                "wearable_steps_change_pct",
                "wearable_hrv_change_pct",
                "wearable_sleep_change_pct"
            ]]
        )

        weekly_obs = get_weekly_observation_df(df)

        if not weekly_obs.empty:
            st.markdown("**Ouderobservaties per week**")
            st.bar_chart(weekly_obs.set_index("year_week")[["ouder_afwijkend_gedrag"]])

            st.markdown("**Leerkrachtscore per week**")
            st.line_chart(weekly_obs.set_index("year_week")[["leerkracht_score"]])

with tab3:
    st.subheader("Interventie-advies")

    if df.empty:
        st.info("Nog geen data beschikbaar.")
    else:
        metrics = compute_metrics(df)
        results = intervention_rules(metrics)
        summary = summarize_interventions(results)

        for intervention, info in summary.items():
            if info["advice"]:
                st.warning(f"{intervention}: overwegen ({info['passed']}/{info['total']} criteria gehaald)")
            else:
                st.success(f"{intervention}: nog niet duidelijk geïndiceerd ({info['passed']}/{info['total']} criteria gehaald)")

            with st.expander(f"Details — {intervention}"):
                details_df = pd.DataFrame({
                    "Criterium": list(results[intervention].keys()),
                    "Gehaald": ["Ja" if v else "Nee" for v in results[intervention].values()]
                })
                st.dataframe(details_df, use_container_width=True)

with tab4:
    st.subheader("Ruwe data")

    if df.empty:
        st.info("Nog geen data opgeslagen.")
    else:
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

        csv = df.copy()
        csv["date"] = csv["date"].astype(str)
        st.download_button(
            label="Download CSV",
            data=csv.to_csv(index=False).encode("utf-8"),
            file_name="duco_daily_log.csv",
            mime="text/csv"
        )

        if st.button("Verwijder alle data"):
            empty_df = init_dataframe()
            save_data(empty_df)
            st.success("Alle data verwijderd. Herlaad de pagina.")