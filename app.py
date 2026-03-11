import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from pathlib import Path

st.set_page_config(
    page_title="CreditScore · Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

:root {
  --bg-base:       #0d0f14;
  --bg-surface:    #13161e;
  --bg-elevated:   #1a1e28;
  --bg-input:      #1e2230;
  --border-sub:    #252a38;
  --border-main:   #2e3448;
  --gold:          #c9a84c;
  --gold-light:    #e2c47a;
  --gold-dim:      rgba(201,168,76,0.18);
  --gold-glow:     rgba(201,168,76,0.08);
  --text-primary:  #f0ede6;
  --text-secondary:#8e95a8;
  --text-muted:    #5a6070;
  --red:           #e05c5c;
  --amber:         #d4943a;
  --green:         #4caf82;
  --radius-sm:     8px;
  --radius-md:     14px;
  --radius-lg:     20px;
}

html, body,
.stApp,
.stApp > div,
section[data-testid="stAppViewContainer"],
.main,
.main .block-container {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'Barlow', sans-serif !important;
}

.main .block-container {
  padding-top: 2rem !important;
  max-width: 1120px !important;
}

[data-testid="stSidebar"] { display: none !important; }

p, span, li, td, th, div, label {
  font-family: 'Barlow', sans-serif !important;
}

/* HERO */
.cv-hero {
  position: relative;
  padding: 52px 56px 48px;
  margin-bottom: 34px;
  background: var(--bg-surface);
  border: 1px solid var(--border-main);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.cv-hero::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 90% 20%, rgba(201,168,76,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(201,168,76,0.04) 0%, transparent 50%);
  pointer-events: none;
}
.cv-hero::after {
  content: '';
  position: absolute;
  top: 0; left: 56px; right: 56px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: 0.6;
}
.cv-eyebrow {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.70rem; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--gold) !important;
  margin: 0 0 18px 0;
  display: flex; align-items: center; gap: 10px;
}
.cv-eyebrow::before {
  content: '';
  display: inline-block;
  width: 24px; height: 1px;
  background: var(--gold);
  opacity: 0.7;
}
.cv-hero-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 3.2rem; font-weight: 700;
  color: var(--text-primary) !important;
  margin: 0 0 6px 0; line-height: 1.05;
  letter-spacing: -0.02em;
}
.cv-hero-title span { color: var(--gold) !important; font-style: italic; }
.cv-hero-sub {
  font-size: 1rem; font-weight: 300;
  color: var(--text-secondary) !important;
  max-width: 620px; line-height: 1.7;
  margin: 16px 0 0 0;
}
.cv-hero-decor {
  position: absolute; right: 56px; top: 50%;
  transform: translateY(-50%);
  font-size: 7rem; line-height: 1;
  color: var(--gold) !important;
  opacity: 0.06;
  font-family: 'Playfair Display', serif !important;
  font-weight: 700;
  pointer-events: none;
  user-select: none;
}

/* SECTION */
.cv-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-main);
  border-radius: var(--radius-lg);
  padding: 26px 26px 20px;
  margin-bottom: 18px;
}
.sec-header {
  display: flex; align-items: flex-start; gap: 14px;
  margin-bottom: 18px; padding-bottom: 14px;
  border-bottom: 1px solid var(--border-sub);
}
.sec-num {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--gold) !important;
  background: var(--gold-dim);
  border: 1px solid rgba(201,168,76,0.3);
  border-radius: 4px;
  padding: 4px 8px;
  margin-top: 3px;
  flex-shrink: 0;
}
.sec-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.18rem; font-weight: 600;
  color: var(--text-primary) !important;
  margin: 0 0 4px 0;
}
.sec-desc {
  font-size: 0.84rem;
  color: var(--text-muted) !important;
  margin: 0;
}

/* FORM */
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stTextInput label {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--text-secondary) !important;
  margin-bottom: 6px !important;
}

div[data-testid="column"] {
  padding-bottom: 0.35rem !important;
}

.stNumberInput input {
  background: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
}
.stNumberInput input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-glow) !important;
  outline: none !important;
}
.stNumberInput button {
  background: var(--bg-elevated) !important;
  color: var(--gold) !important;
  border-color: var(--border-main) !important;
}
.stNumberInput button:hover {
  background: var(--gold-dim) !important;
}

.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stMultiSelect > div > div {
  background: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

.stSelectbox [data-baseweb="select"] span,
.stMultiSelect input {
  color: var(--text-primary) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {
  background: var(--gold-dim) !important;
  color: var(--gold-light) !important;
}

.stMultiSelect [data-baseweb="tag"] {
  background: var(--gold-dim) !important;
  border: 1px solid rgba(201,168,76,0.35) !important;
  border-radius: 5px !important;
}
.stMultiSelect [data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"] button {
  color: var(--gold-light) !important;
}

.cv-hint {
  font-size: 0.76rem;
  color: var(--text-muted) !important;
  margin: 4px 0 4px 2px;
  font-style: italic;
}

/* BUTTON */
.stFormSubmitButton > button {
  background: linear-gradient(135deg, #b8922a 0%, var(--gold) 50%, #d4a84c 100%) !important;
  color: #0d0f14 !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  padding: 15px 40px !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
  width: 100% !important;
  box-shadow: 0 4px 24px rgba(201,168,76,0.25) !important;
}
.stFormSubmitButton > button:hover {
  box-shadow: 0 8px 32px rgba(201,168,76,0.40) !important;
  transform: translateY(-1px) !important;
}

/* RESULT */
.cv-result {
  padding: 34px 40px;
  border-radius: var(--radius-lg);
  margin: 30px 0 20px;
  position: relative; overflow: hidden;
}
.cv-result::after {
  content: '';
  position: absolute; top: 0; left: 40px; right: 40px; height: 1px;
}
.cv-result-bad {
  background: linear-gradient(135deg, #1f1015 0%, #1a1014 100%);
  border: 1px solid rgba(224,92,92,0.35);
}
.cv-result-bad::after { background: linear-gradient(90deg,transparent,#e05c5c,transparent); }

.cv-result-standard {
  background: linear-gradient(135deg, #1a1508 0%, #161208 100%);
  border: 1px solid rgba(212,148,58,0.35);
}
.cv-result-standard::after { background: linear-gradient(90deg,transparent,#d4943a,transparent); }

.cv-result-good {
  background: linear-gradient(135deg, #0b1712 0%, #091410 100%);
  border: 1px solid rgba(76,175,130,0.35);
}
.cv-result-good::after { background: linear-gradient(90deg,transparent,#4caf82,transparent); }

.cv-result-eyebrow {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.20em; text-transform: uppercase;
  margin: 0 0 12px 0;
}
.cv-result-verdict {
  font-family: 'Playfair Display', serif !important;
  font-size: 2.5rem; font-weight: 700;
  margin: 0 0 8px 0; line-height: 1.1;
}
.cv-result-conf {
  font-size: 0.9rem;
  color: var(--text-secondary) !important;
  margin: 0;
}
.cv-result-conf strong {
  color: var(--text-primary) !important;
}

.cv-probs-title {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--text-muted) !important;
  margin: 28px 0 16px;
}
.cv-prob-row {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 13px;
}
.cv-prob-name {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.80rem; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--text-secondary) !important;
  width: 190px; flex-shrink: 0;
}
.cv-prob-track {
  flex: 1; height: 6px;
  background: var(--border-sub);
  border-radius: 100px; overflow: hidden;
}
.cv-prob-fill { height: 100%; border-radius: 100px; }
.cv-prob-pct {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.85rem; font-weight: 700;
  color: var(--text-primary) !important;
  width: 48px; text-align: right;
}

[data-testid="metric-container"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-md) !important;
  padding: 18px 20px !important;
}

[data-testid="stExpander"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-md) !important;
}

[data-testid="stExpander"] summary {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--gold) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

[data-testid="stAlert"] {
  background: rgba(224,92,92,0.10) !important;
  border: 1px solid rgba(224,92,92,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: #f4a0a0 !important;
}

.cv-footer {
  margin-top: 52px;
  padding: 24px 0 14px;
  border-top: 1px solid var(--border-sub);
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 12px;
}
.cv-footer-brand {
  font-family: 'Playfair Display', serif !important;
  font-size: 0.95rem; font-weight: 600;
  color: var(--gold) !important;
}
.cv-footer-note {
  font-size: 0.74rem;
  color: var(--text-muted) !important;
}
.cv-footer-note code {
  color: var(--gold) !important;
  background: var(--gold-dim);
  padding: 1px 6px;
  border-radius: 4px;
}

/* responsive */
@media (max-width: 900px) {
  .cv-hero { padding: 36px 28px; }
  .cv-hero-decor { display: none; }
  .cv-hero-title { font-size: 2.4rem; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════
DATA_URL = "https://github.com/adiacla/bigdata/raw/master/riesgo.xlsx"
MODEL_PATH = "modelo_ann.keras"
SCALER_PATH = "scaler_ann.pkl"
PCA_PATH = "pca_ann.pkl"

TARGET_META = {
    0: dict(
        label="Bad",
        sublabel="Alto Riesgo",
        css="cv-result-bad",
        color="#e05c5c",
        bar="#e05c5c",
    ),
    1: dict(
        label="Standard",
        sublabel="Riesgo Medio",
        css="cv-result-standard",
        color="#d4943a",
        bar="#d4943a",
    ),
    2: dict(
        label="Good",
        sublabel="Bajo Riesgo",
        css="cv-result-good",
        color="#4caf82",
        bar="#4caf82",
    ),
}


# ══════════════════════════════════════════════════════════════════
#  CARGA
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    return model, scaler, pca


@st.cache_data
def load_reference_data():
    df_raw = pd.read_excel(DATA_URL)

    all_loan_types = sorted(
        {
            item.strip()
            for val in df_raw["Type_of_Loan"].dropna().astype(str)
            for item in val.split(",")
            if item.strip() and item.strip().lower() != "not specified"
        }
    )

    df = df_raw.drop(columns=["Customer_ID", "Name", "SSN"]).copy()
    df["Type_of_Loan"] = df["Type_of_Loan"].fillna("Not Specified")
    df["Num_of_Loan_Types"] = df["Type_of_Loan"].apply(
        lambda x: (
            0
            if pd.isna(x) or str(x).strip() == "Not Specified"
            else len(str(x).split(","))
        )
    )
    df = df.drop(columns=["Type_of_Loan"])
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "Yes")

    cat_cols = [
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
    ]

    encoders = {}
    cat_opts = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        cat_opts[col] = list(le.classes_)

    X = df.drop(columns=["Credit_Score"])

    num_defaults = {
        c: float(pd.to_numeric(X[c], errors="coerce").median())
        for c in X.columns
        if c not in cat_cols
    }

    return dict(
        feature_columns=X.columns.tolist(),
        categorical_cols=cat_cols,
        encoders=encoders,
        category_options=cat_opts,
        numeric_defaults=num_defaults,
        loan_type_options=all_loan_types,
    )


def build_feature_row(inputs, meta):
    row = pd.DataFrame([inputs])
    for col in meta["categorical_cols"]:
        row[col] = meta["encoders"][col].transform(row[col].astype(str))
    return row[meta["feature_columns"]]


def nd(key, fallback=0.0):
    return meta["numeric_defaults"].get(key, fallback)


def section_open(num, title, desc):
    st.markdown(
        f"""
        <div class="cv-section">
          <div class="sec-header">
            <span class="sec-num">{num}</span>
            <div>
              <p class="sec-title">{title}</p>
              <p class="sec-desc">{desc}</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="cv-hero">
  <p class="cv-eyebrow">Risk Intelligence Platform</p>
  <h1 class="cv-hero-title">Credit<span>Score</span></h1>
  <p class="cv-hero-sub">
    Sistema de evaluación crediticia basado en una Red Neuronal Artificial con reducción
    de dimensionalidad PCA. Ingresa el perfil financiero del cliente para obtener su
    clasificación de riesgo en tiempo real.
  </p>
  <div class="cv-hero-decor">◆</div>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
#  VALIDAR ARCHIVOS
# ══════════════════════════════════════════════════════════════════
missing = [p for p in [MODEL_PATH, SCALER_PATH, PCA_PATH] if not Path(p).exists()]
if missing:
    st.error(f"Faltan artefactos del modelo: {', '.join(missing)}")
    st.stop()

model, scaler, pca = load_artifacts()
meta = load_reference_data()

# ══════════════════════════════════════════════════════════════════
#  FORMULARIO
# ══════════════════════════════════════════════════════════════════
with st.form("cv_form"):

    section_open(
        "01",
        "Perfil personal y laboral",
        "Datos de edad, ocupación y patrón general de pago.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        age = st.number_input(
            "Edad — Age",
            min_value=18,
            max_value=100,
            value=int(round(nd("Age", 33))),
            help="Edad actual del cliente en años completos.",
        )
        occupation = st.selectbox(
            "Ocupación — Occupation",
            meta["category_options"]["Occupation"],
            help="Actividad principal del cliente.",
        )
    with c2:
        payment_behaviour = st.selectbox(
            "Comportamiento de pago — Payment_Behaviour",
            meta["category_options"]["Payment_Behaviour"],
            help="Patrón de pago registrado históricamente.",
        )
        payment_of_min_amount = st.selectbox(
            "¿Realiza el pago mínimo? — Payment_of_Min_Amount",
            meta["category_options"]["Payment_of_Min_Amount"],
            help="Indica si cumple al menos con el pago mínimo mensual.",
        )
    section_close()

    section_open(
        "02",
        "Ingresos y liquidez",
        "Capacidad de ingreso, saldo mensual e inversión del cliente.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        annual_income = st.number_input(
            "Ingreso anual $ — Annual_Income",
            min_value=0.0,
            value=nd("Annual_Income", 37000.0),
            step=1000.0,
        )
        monthly_salary = st.number_input(
            "Salario mensual neto $ — Monthly_Inhand_Salary",
            min_value=0.0,
            value=nd("Monthly_Inhand_Salary", 3097.0),
            step=100.0,
        )
        monthly_balance = st.number_input(
            "Balance mensual $ — Monthly_Balance",
            value=nd("Monthly_Balance", 338.0),
            step=50.0,
        )
    with c2:
        amount_invested_monthly = st.number_input(
            "Inversión mensual $ — Amount_invested_monthly",
            min_value=0.0,
            value=nd("Amount_invested_monthly", 152.0),
            step=50.0,
        )
        total_emi_per_month = st.number_input(
            "EMI total mensual $ — Total_EMI_per_month",
            min_value=0.0,
            value=nd("Total_EMI_per_month", 68.0),
            step=50.0,
        )
    section_close()

    section_open(
        "03", "Cuentas y tarjetas", "Productos bancarios activos y mezcla crediticia."
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        num_bank_accounts = st.number_input(
            "Nº cuentas bancarias — Num_Bank_Accounts",
            min_value=0,
            value=int(round(nd("Num_Bank_Accounts", 5))),
        )
        num_credit_card = st.number_input(
            "Nº tarjetas de crédito — Num_Credit_Card",
            min_value=0,
            value=int(round(nd("Num_Credit_Card", 5))),
        )
    with c2:
        credit_mix = st.selectbox(
            "Mezcla de crédito — Credit_Mix",
            meta["category_options"]["Credit_Mix"],
        )
        num_credit_inquiries = st.number_input(
            "Consultas al buró — Num_Credit_Inquiries",
            min_value=0.0,
            value=nd("Num_Credit_Inquiries", 5.0),
            step=1.0,
        )
    section_close()

    section_open(
        "04",
        "Préstamos y deuda activa",
        "Cantidad, diversidad de préstamos y deuda vigente.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        num_of_loan = st.number_input(
            "Nº de préstamos — Num_of_Loan",
            min_value=0,
            value=int(round(nd("Num_of_Loan", 3))),
        )
        interest_rate = st.number_input(
            "Tasa de interés % — Interest_Rate",
            min_value=0.0,
            value=nd("Interest_Rate", 13.0),
            step=0.1,
        )
        outstanding_debt = st.number_input(
            "Deuda pendiente $ — Outstanding_Debt",
            min_value=0.0,
            value=nd("Outstanding_Debt", 1166.0),
            step=100.0,
        )
    with c2:
        selected_loan_types = st.multiselect(
            "Tipos de préstamo — Type_of_Loan",
            options=meta["loan_type_options"],
            help="La app convierte esta selección al conteo Num_of_Loan_Types.",
        )
        st.markdown(
            '<p class="cv-hint">El modelo no usa el texto del préstamo directamente; usa el número de tipos seleccionados.</p>',
            unsafe_allow_html=True,
        )
    section_close()

    section_open(
        "05",
        "Historial y comportamiento crediticio",
        "Uso de crédito, retrasos, antigüedad e historial de pagos.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        credit_history_age = st.number_input(
            "Antigüedad del historial — Credit_History_Age",
            min_value=0.0,
            value=nd("Credit_History_Age", 18.0),
            step=1.0,
        )
        credit_utilization_ratio = st.number_input(
            "Utilización de crédito % — Credit_Utilization_Ratio",
            min_value=0.0,
            value=nd("Credit_Utilization_Ratio", 32.0),
            step=0.1,
        )
        changed_credit_limit = st.number_input(
            "Cambio en límite de crédito — Changed_Credit_Limit",
            value=nd("Changed_Credit_Limit", 9.0),
            step=1.0,
        )
    with c2:
        delay_from_due_date = st.number_input(
            "Retraso promedio (días) — Delay_from_due_date",
            value=nd("Delay_from_due_date", 18.0),
            step=1.0,
        )
        num_delayed_payment = st.number_input(
            "Nº pagos atrasados — Num_of_Delayed_Payment",
            min_value=0.0,
            value=nd("Num_of_Delayed_Payment", 14.0),
            step=1.0,
        )
    section_close()

    submitted = st.form_submit_button("◆  Ejecutar análisis crediticio")

# ══════════════════════════════════════════════════════════════════
#  RESULTADO
# ══════════════════════════════════════════════════════════════════
if submitted:
    inputs = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
        "Num_of_Loan_Types": len(selected_loan_types),
    }

    row = build_feature_row(inputs, meta)
    row_scaled = scaler.transform(row)
    row_pca = pca.transform(row_scaled)
    probs = model.predict(row_pca, verbose=0)[0]
    cls = int(np.argmax(probs))
    m = TARGET_META[cls]

    st.markdown(
        f"""
        <div class="cv-result {m['css']}">
          <p class="cv-result-eyebrow" style="color:{m['color']};">◆ Clasificación de riesgo crediticio</p>
          <p class="cv-result-verdict" style="color:{m['color']};">{m['label']}</p>
          <p class="cv-result-conf">
            Categoría: <strong>{m['sublabel']}</strong>
            &nbsp;·&nbsp;
            Confianza del modelo: <strong>{probs[cls]*100:.1f}%</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        m2 = TARGET_META[i]
        col.metric(f"{m2['label']} · {m2['sublabel']}", f"{probs[i]*100:.1f}%")

    st.markdown(
        '<p class="cv-probs-title">Distribución de probabilidad por categoría</p>',
        unsafe_allow_html=True,
    )
    for i in range(3):
        m2 = TARGET_META[i]
        pct = probs[i] * 100
        st.markdown(
            f"""
            <div class="cv-prob-row">
              <span class="cv-prob-name">{m2['label']} — {m2['sublabel']}</span>
              <div class="cv-prob-track">
                <div class="cv-prob-fill" style="width:{pct:.1f}%;background:{m2['bar']};"></div>
              </div>
              <span class="cv-prob-pct">{pct:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Datos técnicos del análisis"):
        st.write("**Parámetros de entrada**")
        st.json(inputs)
        st.write("**Vector de características procesado**")
        st.dataframe(row, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="cv-footer">
  <span class="cv-footer-brand">◆ CreditScore</span>
  <span class="cv-footer-note">
    Powered by ANN + PCA · Requiere
    <code>modelo_ann.keras</code>
    <code>scaler_ann.pkl</code>
    <code>pca_ann.pkl</code>
  </span>
</div>
""",
    unsafe_allow_html=True,
)
