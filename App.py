from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Duco Flow App", page_icon="🧩", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "duco_daily_log.csv"


# =========================
# Defaults and storage
# =========================
DEFAULT_COLUMNS = [
    "date",
    "stoplight_red_pct",
    "stoplight_orange_pct",
    "stoplight_green_pct",
    "questionnaire_score",
    "questionnaire_abnormal",
    "parent_abnormal_count",
    "teacher_score",
    "wearable_steps_drop_pct",
    "wearable_hrv_drop_pct",
    "wearable_sleep_drop_pct",
    "wearable_school_activity_drop_pct",
    "school_fatigue_pattern",
    "psychologist_first_depressive_feature",
    "notes",
]


@dataclass
class RuleResult:
    label: str
    passed: bool
    detail: str


@dataclass
class InterventionResult:
    name: str
    domain: str
    rules: List[RuleResult]
    description: str

    @property
    def score(self) -> Tuple[int, int]:
        passed = sum(rule.passed for rule in self.rules)
        return passed, len(self.rules)

    @property
    def all_passed(self) -> bool:
        return all(rule.passed for rule in self.rules)

    @property
    def any_passed(self) -> bool:
        return any(rule.passed for rule in self.rules)


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    df = pd.read_csv(DATA_FILE)
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[DEFAULT_COLUMNS].copy()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df


def save_data(df: pd.DataFrame) -> None:
    export_df = df.copy()
    if not export_df.empty:
        export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
    export_df.to_csv(DATA_FILE, index=False)


# =========================
# Helpers
# =========================
def check(label: str, condition: bool, detail: str) -> RuleResult:
    return RuleResult(label=label, passed=bool(condition), detail=detail)


def get_baseline(series: pd.Series, baseline_days: int = 14, skip_recent: int = 14, higher_is_worse: bool = True) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    if len(clean) <= skip_recent:
        reference = clean.iloc[: min(len(clean), baseline_days)]
    else:
        reference = clean.iloc[-(baseline_days + skip_recent) : -skip_recent]
    if reference.empty:
        reference = clean.iloc[: min(len(clean), baseline_days)]
    if reference.empty:
        return None
    return float(reference.mean())


def pct_change(current: float, baseline: float | None) -> float:
    if baseline in (None, 0) or pd.isna(baseline):
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def window_stats(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cutoff = df["date"].max() - pd.Timedelta(days=days - 1)
    return df[df["date"] >= cutoff].copy()


# =========================
# Logic based on daily data
# =========================
def evaluate_from_daily_data(df: pd.DataFrame) -> Tuple[List[InterventionResult], dict]:
    if df.empty:
        return [], {}

    df = df.sort_values("date").copy()
    latest = df.iloc[-1]
    week = window_stats(df, 7)
    two_weeks = window_stats(df, 14)
    four_weeks = window_stats(df, 28)

    baseline_red = get_baseline(df["stoplight_red_pct"], baseline_days=14, skip_recent=14)
    baseline_steps = get_baseline(df["wearable_steps_drop_pct"], baseline_days=14, skip_recent=14)
    baseline_hrv = get_baseline(df["wearable_hrv_drop_pct"], baseline_days=14, skip_recent=14)
    baseline_sleep = get_baseline(df["wearable_sleep_drop_pct"], baseline_days=14, skip_recent=14)
    baseline_school = get_baseline(df["wearable_school_activity_drop_pct"], baseline_days=14, skip_recent=14)
    baseline_questionnaire = get_baseline(df["questionnaire_score"], baseline_days=14, skip_recent=14)

    current_red = float(week["stoplight_red_pct"].mean()) if not week.empty else 0.0
    red_change_pct = pct_change(current_red, baseline_red)
    total_red_week_now = float(week["stoplight_red_pct"].sum()) if not week.empty else 0.0
    total_red_week_baseline = (baseline_red or 0.0) * max(len(week), 1)
    total_red_week_change_pct = pct_change(total_red_week_now, total_red_week_baseline)

    questionnaire_abnormal_7d = int(pd.to_numeric(week["questionnaire_abnormal"], errors="coerce").fillna(0).sum()) if not week.empty else 0
    questionnaire_abnormal_14d = int(pd.to_numeric(two_weeks["questionnaire_abnormal"], errors="coerce").fillna(0).sum()) if not two_weeks.empty else 0
    parent_abnormal_7d = int(pd.to_numeric(week["parent_abnormal_count"], errors="coerce").fillna(0).sum()) if not week.empty else 0
    parent_abnormal_14d = int(pd.to_numeric(two_weeks["parent_abnormal_count"], errors="coerce").fillna(0).sum()) if not two_weeks.empty else 0

    questionnaire_now = float(week["questionnaire_score"].mean()) if not week.empty else 0.0
    questionnaire_increase_pct = 0.0
    if baseline_questionnaire not in (None, 0) and pd.notna(baseline_questionnaire):
        # lower score is worse, so reverse sign
        questionnaire_increase_pct = ((baseline_questionnaire - questionnaire_now) / baseline_questionnaire) * 100.0

    wearable_steps_now = float(week["wearable_steps_drop_pct"].mean()) if not week.empty else 0.0
    wearable_hrv_now = float(week["wearable_hrv_drop_pct"].mean()) if not week.empty else 0.0
    wearable_sleep_now = float(week["wearable_sleep_drop_pct"].mean()) if not week.empty else 0.0
    wearable_school_now = float(four_weeks["wearable_school_activity_drop_pct"].mean()) if not four_weeks.empty else 0.0

    school_fatigue_pattern = bool(week["school_fatigue_pattern"].fillna(False).astype(bool).any()) if not week.empty else False
    psychologist_feature = bool(two_weeks["psychologist_first_depressive_feature"].fillna(False).astype(bool).any()) if not two_weeks.empty else False
    teacher_score_28d = float(pd.to_numeric(four_weeks["teacher_score"], errors="coerce").dropna().mean()) if not four_weeks.empty else 10.0
    questionnaire_borderline_or_abnormal = questionnaire_abnormal_7d >= 1

    summaries = {
        "baseline_red": baseline_red,
        "current_red_week": current_red,
        "red_change_pct": red_change_pct,
        "questionnaire_increase_pct": questionnaire_increase_pct,
        "questionnaire_abnormal_7d": questionnaire_abnormal_7d,
        "questionnaire_abnormal_14d": questionnaire_abnormal_14d,
        "parent_abnormal_7d": parent_abnormal_7d,
        "parent_abnormal_14d": parent_abnormal_14d,
        "teacher_score_28d": teacher_score_28d,
        "wearable_steps_now": wearable_steps_now,
        "wearable_hrv_now": wearable_hrv_now,
        "wearable_sleep_now": wearable_sleep_now,
        "wearable_school_now": wearable_school_now,
    }

    results: List[InterventionResult] = []

    results.append(
        InterventionResult(
            name="Add-on programma",
            domain="Psychosociaal / school",
            description="Extra aanpassing binnen het bestaande schoolprogramma.",
            rules=[
                check(
                    "Stoplichtblokje",
                    (red_change_pct >= 5) or (total_red_week_change_pct > 30),
                    f"Weekgemiddelde rood {current_red:.1f}% | Verandering t.o.v. baseline {red_change_pct:.1f}%",
                ),
                check(
                    "Vragenlijst",
                    questionnaire_borderline_or_abnormal or questionnaire_increase_pct >= 20,
                    f"Abnormaal in 7 dagen: {questionnaire_abnormal_7d} | Daling score t.o.v. baseline: {questionnaire_increase_pct:.1f}%",
                ),
                check(
                    "Wearable",
                    (20 <= wearable_steps_now <= 30) and (wearable_hrv_now >= 20) and (wearable_sleep_now >= 15),
                    f"Stappen ↓ {wearable_steps_now:.1f}% | HRV ↓ {wearable_hrv_now:.1f}% | Slaap ↓ {wearable_sleep_now:.1f}%",
                ),
            ],
        )
    )

    results.append(
        InterventionResult(
            name="Online game community",
            domain="Psychosociaal / sociaal",
            description="Veilige online omgeving voor sociale inclusie.",
            rules=[
                check("Stoplichtblokje", red_change_pct >= 5, f"Weekgemiddelde rood verandering: {red_change_pct:.1f}%"),
                check(
                    "Wearable",
                    wearable_hrv_now >= 10 and wearable_sleep_now >= 15,
                    f"HRV ↓ {wearable_hrv_now:.1f}% | Slaap ↓ {wearable_sleep_now:.1f}%",
                ),
                check(
                    "Observatie ouders",
                    parent_abnormal_7d >= 5,
                    f"Opgetelde abnormale gedragingen in 7 dagen: {parent_abnormal_7d}",
                ),
            ],
        )
    )

    results.append(
        InterventionResult(
            name="Muziektherapie",
            domain="Psychosociaal / thuis-klinisch",
            description="Actieve of receptieve muziektherapie.",
            rules=[
                check("Stoplichtblokje", current_red >= 20, f"Weekgemiddelde rood: {current_red:.1f}%"),
                check("Wearable", wearable_hrv_now >= 20 and wearable_sleep_now >= 15, f"HRV ↓ {wearable_hrv_now:.1f}% | Slaap ↓ {wearable_sleep_now:.1f}%"),
                check("Vragenlijst", questionnaire_abnormal_7d >= 3 or questionnaire_increase_pct >= 30, f"Abnormaal in 7 dagen: {questionnaire_abnormal_7d} | Daling score: {questionnaire_increase_pct:.1f}%"),
                check("Observatie ouders", parent_abnormal_7d >= 10, f"Opgeteld in 7 dagen: {parent_abnormal_7d}"),
            ],
        )
    )

    results.append(
        InterventionResult(
            name="Hulpkracht in de klas",
            domain="Psychosociaal / school",
            description="Extra ondersteuning in de klas.",
            rules=[
                check("Stoplichtblokje", 20 <= red_change_pct <= 30, f"Rood verandering t.o.v. baseline: {red_change_pct:.1f}%"),
                check("Wearable", wearable_school_now >= 20 and school_fatigue_pattern, f"Schoolactiviteit ↓ {wearable_school_now:.1f}% | Vermoeidheidspatroon: {school_fatigue_pattern}"),
                check("Observatie leerkracht", teacher_score_28d < 5, f"Gemiddelde leerkrachtscore in 28 dagen: {teacher_score_28d:.1f}/10"),
                check("Kinderpsycholoog", psychologist_feature, f"Eerste depressieve kenmerken aanwezig: {psychologist_feature}"),
            ],
        )
    )

    results.append(
        InterventionResult(
            name="CGT",
            domain="Psychosociaal / klinisch",
            description="Cognitieve gedragstherapie.",
            rules=[
                check("Stoplichtblokje", current_red >= 20, f"Weekgemiddelde rood: {current_red:.1f}%"),
                check("Wearable", (20 <= wearable_steps_now <= 30) and (wearable_hrv_now >= 20) and (wearable_sleep_now >= 20), f"Stappen ↓ {wearable_steps_now:.1f}% | HRV ↓ {wearable_hrv_now:.1f}% | Slaap ↓ {wearable_sleep_now:.1f}%"),
                check("Vragenlijst", questionnaire_abnormal_14d >= 3 or questionnaire_increase_pct >= 30, f"Abnormaal in 14 dagen: {questionnaire_abnormal_14d} | Daling score: {questionnaire_increase_pct:.1f}%"),
                check("Observatie ouders", parent_abnormal_14d >= 20, f"Opgeteld in 14 dagen: {parent_abnormal_14d}"),
                check("Kinderpsycholoog", psychologist_feature, f"Eerste depressieve kenmerken aanwezig: {psychologist_feature}"),
            ],
        )
    )

    return results, summaries


# =========================
# UI
# =========================
st.title("🧩 Duco Duchenne – dagelijkse registratie app")
st.caption("Je vult nu per dag metingen in. De app slaat die op in een CSV-bestand en rekent daar automatisch trends en interventies uit.")

mode = st.sidebar.radio("Beslislogica", ["Strict (alle criteria)", "Signalering (ook partiële matches)"], index=0)
st.sidebar.write(f"Opslaglocatie: `{DATA_FILE}`")

if "save_success" not in st.session_state:
    st.session_state.save_success = False

log_tab, dashboard_tab, data_tab = st.tabs(["Dagregistratie", "Dashboard & advies", "Data beheren"])

df = load_data()

with log_tab:
    st.subheader("Nieuwe dag invoeren")
    with st.form("daily_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            entry_date = st.date_input("Datum", value=pd.Timestamp.today().date())
            stoplight_red_pct = st.number_input("Stoplicht rood (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            stoplight_orange_pct = st.number_input("Stoplicht oranje (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            stoplight_green_pct = st.number_input("Stoplicht groen (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            teacher_score = st.slider("Leerkrachtscore vandaag (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
            questionnaire_score = st.slider("Emotionele vragenlijst score (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
            questionnaire_abnormal = st.checkbox("Vragenlijst vandaag abnormaal/borderline")

        with col2:
            parent_abnormal_count = st.number_input("Aantal afwijkende gedragingen vandaag", min_value=0, max_value=20, value=0, step=1)
            wearable_steps_drop_pct = st.number_input("Wearable: daling stappen (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            wearable_hrv_drop_pct = st.number_input("Wearable: daling HRV (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            wearable_sleep_drop_pct = st.number_input("Wearable: slechtere slaap (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            wearable_school_activity_drop_pct = st.number_input("Wearable: daling schoolactiviteit (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            school_fatigue_pattern = st.checkbox("Middagdip / vermoeidheidspatroon vandaag")

        with col3:
            psychologist_first_depressive_feature = st.checkbox("Kinderpsycholoog: eerste depressieve kenmerken")
            notes = st.text_area("Notities", placeholder="Bijvoorbeeld: ruzie op school, slecht geslapen, theatermiddag, etc.")

        submitted = st.form_submit_button("Opslaan")

    if submitted:
        total = stoplight_red_pct + stoplight_orange_pct + stoplight_green_pct
        if abs(total - 100.0) > 0.5:
            st.error("De drie stoplichtpercentages moeten samen ongeveer 100% zijn.")
        else:
            new_row = pd.DataFrame([
                {
                    "date": pd.to_datetime(entry_date),
                    "stoplight_red_pct": stoplight_red_pct,
                    "stoplight_orange_pct": stoplight_orange_pct,
                    "stoplight_green_pct": stoplight_green_pct,
                    "questionnaire_score": questionnaire_score,
                    "questionnaire_abnormal": int(questionnaire_abnormal),
                    "parent_abnormal_count": parent_abnormal_count,
                    "teacher_score": teacher_score,
                    "wearable_steps_drop_pct": wearable_steps_drop_pct,
                    "wearable_hrv_drop_pct": wearable_hrv_drop_pct,
                    "wearable_sleep_drop_pct": wearable_sleep_drop_pct,
                    "wearable_school_activity_drop_pct": wearable_school_activity_drop_pct,
                    "school_fatigue_pattern": bool(school_fatigue_pattern),
                    "psychologist_first_depressive_feature": bool(psychologist_first_depressive_feature),
                    "notes": notes,
                }
            ])

            df_no_same_day = df[df["date"] != pd.to_datetime(entry_date)].copy() if not df.empty else df.copy()
            updated_df = pd.concat([df_no_same_day, new_row], ignore_index=True).sort_values("date")
            save_data(updated_df)
            df = load_data()
            st.success(f"Dagregistratie voor {entry_date} opgeslagen.")

with dashboard_tab:
    st.subheader("Dashboard")
    if df.empty:
        st.info("Er is nog geen data opgeslagen. Vul eerst een dagregistratie in.")
    else:
        results, summaries = evaluate_from_daily_data(df)

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Aantal registratiedagen", len(df))
        top2.metric("Gem. rood afgelopen 7 dagen", f"{summaries['current_red_week']:.1f}%")
        top3.metric("Afwijkingen ouders 7 dagen", summaries["parent_abnormal_7d"])
        top4.metric("Leerkrachtscore 28 dagen", f"{summaries['teacher_score_28d']:.1f}/10")

        chart_df = df.sort_values("date").set_index("date")
        st.markdown("### Trends")
        st.line_chart(chart_df[["stoplight_red_pct", "questionnaire_score", "teacher_score"]])
        st.line_chart(chart_df[["wearable_hrv_drop_pct", "wearable_sleep_drop_pct", "wearable_steps_drop_pct"]])

        st.markdown("### Advies vanuit het stroomschema")
        recommended = [r for r in results if r.all_passed] if mode.startswith("Strict") else [r for r in results if r.any_passed]

        if recommended:
            for result in recommended:
                passed, total = result.score
                st.success(f"**{result.name}** • {passed}/{total} criteria behaald")
                st.write(result.description)
                with st.expander(f"Bekijk criteria voor {result.name}"):
                    for rule in result.rules:
                        st.write(f"{'✅' if rule.passed else '❌'} **{rule.label}** — {rule.detail}")
        else:
            st.info("Op basis van de opgeslagen dagdata wordt nog geen interventie volledig geactiveerd.")

        summary_rows = []
        for result in results:
            passed, total = result.score
            summary_rows.append(
                {
                    "Interventie": result.name,
                    "Domein": result.domain,
                    "Behaalde criteria": f"{passed}/{total}",
                    "Strict actief": "Ja" if result.all_passed else "Nee",
                    "Signaal aanwezig": "Ja" if result.any_passed else "Nee",
                }
            )
        st.markdown("### Overzicht interventies")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Opgeslagen dagdata")
    if df.empty:
        st.info("Nog geen opgeslagen data.")
    else:
        display_df = df.copy()
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.date
        st.dataframe(display_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="duco_daily_log.csv", mime="text/csv")

        st.markdown("### Verwijderen")
        dates = sorted(display_df["date"].astype(str).unique(), reverse=True)
        delete_date = st.selectbox("Verwijder registratie van datum", options=dates)
        if st.button("Verwijder geselecteerde dag"):
            new_df = df[df["date"].dt.strftime("%Y-%m-%d") != delete_date].copy()
            save_data(new_df)
            st.success(f"Registratie van {delete_date} verwijderd. Herlaad de pagina even als nodig.")

st.markdown("---")
st.caption("De app slaat dagregistraties lokaal op in `data/duco_daily_log.csv`. In GitHub kun je deze app direct in je repository zetten; de datafile wordt dan automatisch aangemaakt zodra je de eerste dag invoert.")
