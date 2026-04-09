import math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Duco Duchenne Flow App",
    page_icon="🧩",
    layout="wide",
)


# =========================
# Data model
# =========================
@dataclass
class RuleResult:
    label: str
    passed: bool
    detail: str


@dataclass
class InterventionResult:
    name: str
    domain: str
    trigger_mode: str
    rules: List[RuleResult]
    description: str

    @property
    def score(self) -> Tuple[int, int]:
        passed = sum(rule.passed for rule in self.rules)
        total = len(self.rules)
        return passed, total

    @property
    def all_passed(self) -> bool:
        return all(rule.passed for rule in self.rules)

    @property
    def any_passed(self) -> bool:
        return any(rule.passed for rule in self.rules)


# =========================
# Helpers
# =========================
def pct_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def check(label: str, condition: bool, detail: str) -> RuleResult:
    return RuleResult(label=label, passed=bool(condition), detail=detail)


# =========================
# Clinical logic based on uploaded flow / slides
# =========================
def evaluate_interventions(data: dict) -> List[InterventionResult]:
    # Derived values
    stoplight_red_change_pct = pct_change(data["stoplight_red_now"], data["stoplight_red_baseline"])
    stoplight_total_change_pct = pct_change(data["stoplight_total_red_week_now"], data["stoplight_total_red_week_baseline"])

    results: List[InterventionResult] = []

    # 1) Add-on programma
    add_on_rules = [
        check(
            "Stoplichtblokje",
            (stoplight_red_change_pct >= 5 and data["stoplight_duration_days"] >= 7)
            or (stoplight_total_change_pct > 30 and data["stoplight_duration_days"] >= 7),
            f"Rood verandering: {stoplight_red_change_pct:.1f}% | Totale weekverandering: {stoplight_total_change_pct:.1f}% | Duur: {data['stoplight_duration_days']} dagen",
        ),
        check(
            "Vragenlijst",
            data["questionnaire_borderline_or_abnormal"]
            or (data["questionnaire_increase_pct"] >= 20 and data["questionnaire_duration_days"] >= 7),
            f"Borderline/abnormaal: {data['questionnaire_borderline_or_abnormal']} | Toename: {data['questionnaire_increase_pct']:.1f}% | Duur: {data['questionnaire_duration_days']} dagen",
        ),
        check(
            "Wearable",
            (20 <= data["wearable_steps_drop_pct"] <= 30)
            and (data["wearable_hrv_drop_pct"] >= 20)
            and (data["wearable_sleep_drop_pct"] >= 15)
            and (data["wearable_duration_days"] >= 7),
            f"Stappen ↓ {data['wearable_steps_drop_pct']:.1f}% | HRV ↓ {data['wearable_hrv_drop_pct']:.1f}% | Slaap ↓ {data['wearable_sleep_drop_pct']:.1f}% | Duur: {data['wearable_duration_days']} dagen",
        ),
    ]
    results.append(
        InterventionResult(
            name="Add-on programma",
            domain="Psychosociaal / school",
            trigger_mode="strict",
            rules=add_on_rules,
            description="Extra aanpassing binnen het bestaande schoolprogramma, zodat Duco kan meedoen zonder apart traject.",
        )
    )

    # 2) Online game community
    online_rules = [
        check(
            "Stoplichtblokje",
            stoplight_red_change_pct >= 5 and data["stoplight_duration_days"] >= 7,
            f"Rood verandering: {stoplight_red_change_pct:.1f}% | Duur: {data['stoplight_duration_days']} dagen",
        ),
        check(
            "Wearable",
            data["wearable_hrv_drop_pct"] >= 10
            and data["wearable_sleep_drop_pct"] >= 15
            and data["wearable_duration_days"] >= 7,
            f"HRV ↓ {data['wearable_hrv_drop_pct']:.1f}% | Slaap ↓ {data['wearable_sleep_drop_pct']:.1f}% | Duur: {data['wearable_duration_days']} dagen",
        ),
        check(
            "Observatie ouders",
            data["parent_abnormal_count_7d"] >= 5,
            f"Abnormale gedragsveranderingen in 7 dagen: {data['parent_abnormal_count_7d']}",
        ),
    ]
    results.append(
        InterventionResult(
            name="Online game community",
            domain="Psychosociaal / sociaal",
            trigger_mode="strict",
            rules=online_rules,
            description="Veilige online omgeving om sociale inclusie en contact met peers of lotgenoten te bevorderen.",
        )
    )

    # 3) Muziektherapie
    music_rules = [
        check(
            "Stoplichtblokje",
            data["stoplight_red_now"] >= 20 and data["stoplight_duration_days"] >= 14,
            f"Rood nu: {data['stoplight_red_now']:.1f}% | Duur: {data['stoplight_duration_days']} dagen",
        ),
        check(
            "Wearable",
            data["wearable_hrv_drop_pct"] >= 20
            and data["wearable_sleep_drop_pct"] >= 15,
            f"HRV ↓ {data['wearable_hrv_drop_pct']:.1f}% | Slaap ↓ {data['wearable_sleep_drop_pct']:.1f}%",
        ),
        check(
            "Vragenlijst",
            data["questionnaire_abnormal_count_7d"] >= 3
            or (data["questionnaire_increase_pct"] >= 30 and data["questionnaire_duration_days"] >= 7),
            f"Abnormaal in 7 dagen: {data['questionnaire_abnormal_count_7d']} | Toename: {data['questionnaire_increase_pct']:.1f}%",
        ),
        check(
            "Observatie ouders",
            data["parent_abnormal_count_7d"] >= 10,
            f"Abnormale gedragsveranderingen in 7 dagen: {data['parent_abnormal_count_7d']}",
        ),
    ]
    results.append(
        InterventionResult(
            name="Muziektherapie",
            domain="Psychosociaal / thuis-klinisch",
            trigger_mode="strict",
            rules=music_rules,
            description="Actieve of receptieve muziektherapie om stress, spanning en emotionele verwerking te ondersteunen.",
        )
    )

    # 4) Hulpkracht in de klas
    helper_rules = [
        check(
            "Stoplichtblokje",
            20 <= stoplight_red_change_pct <= 30 and data["stoplight_duration_days"] >= 30,
            f"Rood verandering: {stoplight_red_change_pct:.1f}% | Duur: {data['stoplight_duration_days']} dagen",
        ),
        check(
            "Wearable",
            data["wearable_school_activity_drop_pct"] >= 20
            and data["wearable_duration_days"] >= 30
            and data["school_fatigue_pattern"],
            f"Schoolactiviteit ↓ {data['wearable_school_activity_drop_pct']:.1f}% | Middagdip/vermoeidheidspatroon: {data['school_fatigue_pattern']} | Duur: {data['wearable_duration_days']} dagen",
        ),
        check(
            "Observatie leerkracht",
            data["teacher_observation_score_28d"] < 5,
            f"Gemiddelde observatiescore over 28 dagen: {data['teacher_observation_score_28d']:.1f}/10",
        ),
        check(
            "Kinderpsycholoog",
            data["psychologist_first_depressive_feature"],
            f"Eerste kenmerk depressieve stoornis aanwezig: {data['psychologist_first_depressive_feature']}",
        ),
    ]
    results.append(
        InterventionResult(
            name="Hulpkracht in de klas",
            domain="Psychosociaal / school",
            trigger_mode="strict",
            rules=helper_rules,
            description="Extra ondersteuning in de klas voor taakaanpassing, inclusie en ontlasting van de leerkracht.",
        )
    )

    # 5) CGT
    cgt_rules = [
        check(
            "Stoplichtblokje",
            data["stoplight_red_now"] >= 20 and data["stoplight_duration_days"] >= 14,
            f"Rood nu: {data['stoplight_red_now']:.1f}% | Duur: {data['stoplight_duration_days']} dagen",
        ),
        check(
            "Wearable",
            (20 <= data["wearable_steps_drop_pct"] <= 30)
            and (data["wearable_hrv_drop_pct"] >= 20)
            and (data["wearable_sleep_drop_pct"] >= 20)
            and (data["wearable_duration_days"] >= 14),
            f"Stappen ↓ {data['wearable_steps_drop_pct']:.1f}% | HRV ↓ {data['wearable_hrv_drop_pct']:.1f}% | Slaap ↓ {data['wearable_sleep_drop_pct']:.1f}% | Duur: {data['wearable_duration_days']} dagen",
        ),
        check(
            "Vragenlijst",
            data["questionnaire_abnormal_count_14d"] >= 3
            or (data["questionnaire_increase_pct"] >= 30 and data["questionnaire_duration_days"] >= 7),
            f"Abnormaal in 14 dagen: {data['questionnaire_abnormal_count_14d']} | Toename: {data['questionnaire_increase_pct']:.1f}%",
        ),
        check(
            "Observatie ouders",
            data["parent_abnormal_count_14d"] >= 20,
            f"Abnormale gedragsveranderingen in 14 dagen: {data['parent_abnormal_count_14d']}",
        ),
        check(
            "Kinderpsycholoog",
            data["psychologist_first_depressive_feature"],
            f"Eerste kenmerk depressieve stoornis aanwezig: {data['psychologist_first_depressive_feature']}",
        ),
    ]
    results.append(
        InterventionResult(
            name="CGT",
            domain="Psychosociaal / klinisch",
            trigger_mode="strict",
            rules=cgt_rules,
            description="Cognitieve gedragstherapie om negatieve gedachten, emotionele belasting en copingproblemen gericht aan te pakken.",
        )
    )

    return results


# =========================
# UI
# =========================
st.title("🧩 Duco Duchenne – psychosociaal stroomschema")
st.caption(
    "Streamlit prototype gebaseerd op jullie psychosociale flowchart en de uitwerking uit Theme 3 en Theme 4."
)

with st.expander("Wat doet deze app?"):
    st.markdown(
        """
Deze app vertaalt jullie stroomschema naar een **beslissingsondersteunende demo**.

Je vult per meetbron de actuele situatie in:
- stoplichtblokje
- vragenlijst
- observaties van ouders en leerkracht
- wearable
- kinderpsycholoog

Daarna rekent de app uit welke interventies volgens jullie flow het meest passend zijn:
- Add-on programma
- Online game community
- Muziektherapie
- Hulpkracht in de klas
- CGT

De logica is bewust transparant gemaakt, zodat je hem later makkelijk kunt aanpassen aan jullie definitieve eisen.
        """
    )

st.sidebar.header("Instellingen")
mode = st.sidebar.radio(
    "Beslislogica",
    options=["Strict (alle criteria per interventie)", "Signalering (toon ook partiële matches)"],
    index=0,
)
show_flow_table = st.sidebar.checkbox("Toon samenvatting invoer als tabel", value=True)

st.subheader("1. Invoer van metingen")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Stoplichtblokje")
    stoplight_red_baseline = st.number_input("Baseline % rood", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
    stoplight_red_now = st.number_input("Huidig % rood", min_value=0.0, max_value=100.0, value=22.0, step=1.0)
    stoplight_total_red_week_baseline = st.number_input("Baseline totaal roodmomenten/week", min_value=0.0, value=10.0, step=1.0)
    stoplight_total_red_week_now = st.number_input("Huidig totaal roodmomenten/week", min_value=0.0, value=16.0, step=1.0)
    stoplight_duration_days = st.slider("Duur afwijking stoplicht (dagen)", min_value=0, max_value=60, value=14)

    st.markdown("### Observatie leerkracht")
    teacher_observation_score_28d = st.slider("Gemiddelde score over 28 dagen (1-10)", min_value=1.0, max_value=10.0, value=4.5, step=0.5)

with col2:
    st.markdown("### Vragenlijst")
    questionnaire_borderline_or_abnormal = st.checkbox("Minstens één borderline/abnormale score", value=True)
    questionnaire_increase_pct = st.slider("Toename t.o.v. baseline (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
    questionnaire_abnormal_count_7d = st.slider("Aantal abnormale scores in 7 dagen", min_value=0, max_value=14, value=3)
    questionnaire_abnormal_count_14d = st.slider("Aantal abnormale scores in 14 dagen", min_value=0, max_value=28, value=4)
    questionnaire_duration_days = st.slider("Duur afwijking vragenlijst (dagen)", min_value=0, max_value=30, value=10)

    st.markdown("### Observatie ouders")
    parent_abnormal_count_7d = st.slider("Aantal abnormale gedragsveranderingen in 7 dagen", min_value=0, max_value=30, value=10)
    parent_abnormal_count_14d = st.slider("Aantal abnormale gedragsveranderingen in 14 dagen", min_value=0, max_value=50, value=20)

with col3:
    st.markdown("### Wrist wearable")
    wearable_steps_drop_pct = st.slider("Daling stappen (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    wearable_hrv_drop_pct = st.slider("Daling HRV (%)", min_value=0.0, max_value=100.0, value=22.0, step=1.0)
    wearable_sleep_drop_pct = st.slider("Verslechtering slaap (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    wearable_school_activity_drop_pct = st.slider("Daling activiteit tijdens schooluren (%)", min_value=0.0, max_value=100.0, value=24.0, step=1.0)
    wearable_duration_days = st.slider("Duur afwijking wearable (dagen)", min_value=0, max_value=60, value=14)
    school_fatigue_pattern = st.checkbox("Middagdip / vermoeidheidspatroon zichtbaar", value=True)

    st.markdown("### Kinderpsycholoog")
    psychologist_first_depressive_feature = st.checkbox("Eerste kenmerk depressieve stoornis aanwezig", value=True)


data = {
    "stoplight_red_baseline": stoplight_red_baseline,
    "stoplight_red_now": stoplight_red_now,
    "stoplight_total_red_week_baseline": stoplight_total_red_week_baseline,
    "stoplight_total_red_week_now": stoplight_total_red_week_now,
    "stoplight_duration_days": stoplight_duration_days,
    "teacher_observation_score_28d": teacher_observation_score_28d,
    "questionnaire_borderline_or_abnormal": questionnaire_borderline_or_abnormal,
    "questionnaire_increase_pct": questionnaire_increase_pct,
    "questionnaire_abnormal_count_7d": questionnaire_abnormal_count_7d,
    "questionnaire_abnormal_count_14d": questionnaire_abnormal_count_14d,
    "questionnaire_duration_days": questionnaire_duration_days,
    "parent_abnormal_count_7d": parent_abnormal_count_7d,
    "parent_abnormal_count_14d": parent_abnormal_count_14d,
    "wearable_steps_drop_pct": wearable_steps_drop_pct,
    "wearable_hrv_drop_pct": wearable_hrv_drop_pct,
    "wearable_sleep_drop_pct": wearable_sleep_drop_pct,
    "wearable_school_activity_drop_pct": wearable_school_activity_drop_pct,
    "wearable_duration_days": wearable_duration_days,
    "school_fatigue_pattern": school_fatigue_pattern,
    "psychologist_first_depressive_feature": psychologist_first_depressive_feature,
}

results = evaluate_interventions(data)

st.subheader("2. Advies vanuit het stroomschema")

if mode.startswith("Strict"):
    recommended = [r for r in results if r.all_passed]
else:
    recommended = [r for r in results if r.any_passed]

if recommended:
    for result in recommended:
        passed, total = result.score
        st.success(f"**{result.name}**  •  {passed}/{total} criteria behaald")
        st.write(result.description)
        with st.expander(f"Bekijk criteria voor {result.name}"):
            for rule in result.rules:
                icon = "✅" if rule.passed else "❌"
                st.write(f"{icon} **{rule.label}** — {rule.detail}")
else:
    st.info("Op basis van de huidige invoer wordt nog geen interventie volledig geactiveerd.")

st.subheader("3. Overzicht alle interventies")

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

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

with st.expander("Detail per interventie"):
    for result in results:
        passed, total = result.score
        st.markdown(f"### {result.name} ({passed}/{total})")
        for rule in result.rules:
            icon = "✅" if rule.passed else "❌"
            st.write(f"{icon} **{rule.label}:** {rule.detail}")

st.subheader("4. Cascade")
st.markdown(
    """
Volgens jullie cascade lopen de psychosociale interventies van relatief minder invasief naar meer invasief.
Een praktische implementatie kan bijvoorbeeld deze volgorde gebruiken:

1. Add-on programma
2. Online game community
3. Muziektherapie
4. Hulpkracht in de klas
5. CGT

In een echte klinische of onderwijssetting zou je hier nog een **multidisciplinair akkoordmoment** tussen kunnen zetten.
    """
)

cascade_scores = pd.DataFrame(
    {
        "Interventie": [r.name for r in results],
        "Score": [r.score[0] for r in results],
    }
)
st.bar_chart(cascade_scores.set_index("Interventie"))

if show_flow_table:
    st.subheader("5. Samenvatting invoer")
    input_df = pd.DataFrame(
        [
            ("Stoplicht baseline % rood", stoplight_red_baseline),
            ("Stoplicht huidig % rood", stoplight_red_now),
            ("Stoplicht duur (dagen)", stoplight_duration_days),
            ("Vragenlijst toename %", questionnaire_increase_pct),
            ("Abnormaal 7 dagen", questionnaire_abnormal_count_7d),
            ("Abnormaal 14 dagen", questionnaire_abnormal_count_14d),
            ("Ouders afwijkend gedrag 7 dagen", parent_abnormal_count_7d),
            ("Ouders afwijkend gedrag 14 dagen", parent_abnormal_count_14d),
            ("Wearable stappen daling %", wearable_steps_drop_pct),
            ("Wearable HRV daling %", wearable_hrv_drop_pct),
            ("Wearable slaap daling %", wearable_sleep_drop_pct),
            ("Leerkracht score 28 dagen", teacher_observation_score_28d),
        ],
        columns=["Parameter", "Waarde"],
    )
    st.dataframe(input_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Prototype voor jullie GitHub-repository. De drempels en formuleringen staan bewust in Python variabelen, zodat je ze later makkelijk kunt aanpassen."
)
