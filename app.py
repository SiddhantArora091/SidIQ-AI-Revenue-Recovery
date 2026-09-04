import streamlit as st
import pandas as pd
import numpy as np
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SidIQ | Intelligent Revenue Recovery",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    font-size: 18px;
    color: #888888;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.sidiq-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.sidiq-logo {
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
}
.sidiq-brand-title {
    font-size: 30px;
    font-weight: 800;
    line-height: 1;
}
.sidiq-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.sidiq-sidebar-logo {
    width: 27px;
    height: 27px;
}
.sidiq-sidebar-title {
    font-size: 19px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Keep the left navigation clearly visible */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    width: 260px !important;
}

section[data-testid="stSidebar"] > div {
    width: 260px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 8px;
    border-radius: 8px;
    margin-bottom: 3px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="sidiq-brand">\n<svg class="sidiq-logo" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="SidIQ logo">\n  <defs>\n    <linearGradient id="sidBlue" x1="0" y1="0" x2="1" y2="1">\n      <stop offset="0%" stop-color="#1677FF"/>\n      <stop offset="100%" stop-color="#00B7FF"/>\n    </linearGradient>\n  </defs>\n  <path d="M24 4 42 14v20L24 44 6 34V14L24 4Z" fill="none" stroke="url(#sidBlue)" stroke-width="3.2" stroke-linejoin="round"/>\n  <path d="M13 31V22h6v9h-6Zm8 0V17h6v14h-6Zm8 0V13h6v18h-6Z" fill="url(#sidBlue)" opacity=".96"/>\n  <path d="M12 18l8-5 6 4 10-8" fill="none" stroke="#0B67D1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>\n  <path d="m33 9 3-.2-.8 3" fill="none" stroke="#0B67D1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>\n</svg>\n<div class="sidiq-brand-title">SidIQ</div></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Revenue Recovery & Intelligent Intervention Engine'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "SidIQ uses historical transaction and recovery-action "
    "data to predict recovery probability, understand customer "
    "behavior, detect recovery fatigue, and recommend the action "
    "that maximizes expected net recovery."
)


# ============================================================
# LOAD CSV DATA
# ============================================================

@st.cache_data
def load_data():

    transactions = pd.read_csv(
        "transactions.csv"
    )

    recovery_actions = pd.read_csv(
        "recovery_actions.csv"
    )

    return transactions, recovery_actions


try:

    df, actions_df = load_data()

except FileNotFoundError:

    st.error(
        "CSV files not found. Make sure transactions.csv and "
        "recovery_actions.csv are in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()

actions_df.columns = actions_df.columns.str.strip()


numeric_columns = [

    "amount",
    "previous_successful_payments",
    "previous_recovery_success",
    "recovery_attempts",
    "customer_annual_value",
    "recovered"

]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# CUSTOMER HISTORY
# ============================================================

customer_history = (

    df.groupby("customer_id")

    .agg(

        total_transactions=(
            "transaction_id",
            "count"
        ),

        successful_recoveries=(
            "recovered",
            "sum"
        ),

        total_transaction_value=(
            "amount",
            "sum"
        ),

        average_recovery_attempts=(
            "recovery_attempts",
            "mean"
        )

    )

    .reset_index()

)


customer_history[
    "historical_recovery_rate"
] = (

    customer_history[
        "successful_recoveries"
    ]

    /

    customer_history[
        "total_transactions"
    ]

)


customer_history[
    "historical_recovery_rate"
] = (

    customer_history[
        "historical_recovery_rate"
    ].fillna(0)

)


# Merge customer history with transactions

df = df.merge(

    customer_history,

    on="customer_id",

    how="left"

)


# ============================================================
# CUSTOMER SEGMENT
# ============================================================

def get_customer_segment(row):

    recovery_rate = row[
        "historical_recovery_rate"
    ]

    value = row[
        "customer_annual_value"
    ]

    if value >= 75000 and recovery_rate >= 0.50:

        return "⭐ High Value / High Recovery"

    elif value >= 75000:

        return "💎 High Value"

    elif recovery_rate >= 0.50:

        return "🟢 Recovery Friendly"

    elif recovery_rate < 0.20:

        return "🔴 Low Recovery History"

    else:

        return "🟡 Standard"


df["customer_segment"] = df.apply(
    get_customer_segment,
    axis=1
)


# ============================================================
# RECOVERY FATIGUE
# ============================================================

def calculate_fatigue(row):

    attempts = row[
        "recovery_attempts"
    ]

    average_attempts = row[
        "average_recovery_attempts"
    ]

    if attempts >= 6:

        return 100, "CRITICAL"

    elif attempts >= 4:

        return 75, "HIGH"

    elif attempts >= 2:

        return 45, "MEDIUM"

    elif average_attempts >= 3:

        return 30, "MEDIUM"

    else:

        return 10, "LOW"


fatigue_data = df.apply(
    calculate_fatigue,
    axis=1
)


df[
    "recovery_fatigue_score"
] = fatigue_data.apply(
    lambda x: x[0]
)


df[
    "recovery_fatigue_level"
] = fatigue_data.apply(
    lambda x: x[1]
)


# ============================================================
# ML DATA PREPARATION
# ============================================================

ml_df = df.copy()


# Convert categorical columns

ml_df = pd.get_dummies(

    ml_df,

    columns=[
        "payment_method",
        "failure_reason"
    ]

)


base_features = [

    "amount",

    "previous_successful_payments",

    "previous_recovery_success",

    "recovery_attempts",

    "customer_annual_value",

    "historical_recovery_rate",

    "average_recovery_attempts"

]


categorical_features = [

    col

    for col in ml_df.columns

    if (
        col.startswith("payment_method_")
        or
        col.startswith("failure_reason_")
    )

]


features = (
    base_features
    +
    categorical_features
)


X = ml_df[features]

y = ml_df["recovered"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=10,

        min_samples_split=5,

        random_state=42

    )

    model.fit(
        X_train,
        y_train
    )

    return model


model = train_model(
    X_train,
    y_train
)


# ============================================================
# MODEL PREDICTION
# ============================================================

test_predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)


precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
)


# Predict probability for every transaction

df[
    "ml_recovery_probability"
] = model.predict_proba(
    ml_df[features]
)[:, 1]


# ============================================================
# ACTION COSTS
# ============================================================

ACTION_COSTS = {

    "Smart Retry": 3,

    "Delayed Retry": 4,

    "Payment Reminder": 5,

    "Payment Link": 7,

    "Payment Method Update": 8,

    "Human Review": 10

}


# ============================================================
# HISTORICAL ACTION PERFORMANCE
# ============================================================

action_performance = (

    actions_df.groupby(
        "action_taken"
    )

    .agg(

        historical_attempts=(
            "recovered",
            "count"
        ),

        historical_recoveries=(
            "recovered",
            "sum"
        ),

        average_cost=(
            "intervention_cost",
            "mean"
        )

    )

    .reset_index()

)


action_performance[
    "historical_success_rate"
] = (

    action_performance[
        "historical_recoveries"
    ]

    /

    action_performance[
        "historical_attempts"
    ]

)


# ============================================================
# ACTION OPTIMIZER
# ============================================================

def optimize_action(row):

    amount = row["amount"]

    base_probability = row[
        "ml_recovery_probability"
    ]

    failure_reason = row[
        "failure_reason"
    ]

    payment_method = row[
        "payment_method"
    ]

    fatigue_level = row[
        "recovery_fatigue_level"
    ]

    customer_recovery_rate = row[
        "historical_recovery_rate"
    ]


    results = []


    for action in ACTION_COSTS:

        cost = ACTION_COSTS[action]


        # ----------------------------------------------------
        # HISTORICAL ACTION SUCCESS
        # ----------------------------------------------------

        historical_row = action_performance[
            action_performance[
                "action_taken"
            ] == action
        ]


        if len(historical_row) > 0:

            historical_rate = (

                historical_row.iloc[0][
                    "historical_success_rate"
                ]

            )

        else:

            historical_rate = 0.30


        # ----------------------------------------------------
        # COMBINE ML + HISTORICAL ACTION SIGNAL
        # ----------------------------------------------------

        probability = (

            0.65 * base_probability

            +

            0.35 * historical_rate

        )


        # ----------------------------------------------------
        # FAILURE-SPECIFIC INTELLIGENCE
        # ----------------------------------------------------

        if (
            failure_reason
            ==
            "Temporary Bank/Network Error"
        ):

            if action == "Smart Retry":

                probability += 0.12

            elif action == "Delayed Retry":

                probability += 0.05


        elif (
            failure_reason
            ==
            "Insufficient Funds"
        ):

            if action == "Delayed Retry":

                probability += 0.12

            elif action == "Payment Reminder":

                probability += 0.06


        elif (
            failure_reason
            ==
            "Authentication Failed"
        ):

            if action == "Payment Link":

                probability += 0.12

            elif action == "Smart Retry":

                probability -= 0.08


        elif (
            failure_reason
            ==
            "Card Expired"
        ):

            if action == "Payment Method Update":

                probability += 0.18

            elif action == "Smart Retry":

                probability -= 0.15


        # ----------------------------------------------------
        # PAYMENT METHOD SIGNAL
        # ----------------------------------------------------

        if (
            payment_method == "UPI"
            and
            action == "Smart Retry"
        ):

            probability += 0.04


        # ----------------------------------------------------
        # CUSTOMER HISTORY
        # ----------------------------------------------------

        if customer_recovery_rate >= 0.60:

            if action in [
                "Smart Retry",
                "Delayed Retry"
            ]:

                probability += 0.05


        elif customer_recovery_rate <= 0.20:

            if action == "Human Review":

                probability += 0.08


        # ----------------------------------------------------
        # RECOVERY FATIGUE
        # ----------------------------------------------------

        if fatigue_level == "MEDIUM":

            if action == "Smart Retry":

                probability -= 0.05


        elif fatigue_level == "HIGH":

            if action == "Smart Retry":

                probability -= 0.15

            elif action == "Delayed Retry":

                probability -= 0.08

            elif action == "Human Review":

                probability += 0.08


        elif fatigue_level == "CRITICAL":

            if action in [
                "Smart Retry",
                "Delayed Retry"
            ]:

                probability -= 0.25

            elif action == "Human Review":

                probability += 0.15


        probability = float(
            np.clip(
                probability,
                0.01,
                0.98
            )
        )


        # ----------------------------------------------------
        # EXPECTED RECOVERY
        # ----------------------------------------------------

        expected_recovery = (

            amount
            *
            probability

        )


        # ----------------------------------------------------
        # NET RECOVERY
        # ----------------------------------------------------

        expected_net_recovery = (

            expected_recovery
            -
            cost

        )


        results.append({

            "Action":
                action,

            "Probability":
                probability,

            "Expected Recovery":
                expected_recovery,

            "Cost":
                cost,

            "Expected Net Recovery":
                expected_net_recovery

        })


    comparison = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # CRITICAL FATIGUE SAFETY RULE
    # --------------------------------------------------------

    if fatigue_level == "CRITICAL":

        recommended_action = "Human Review"

    else:

        recommended_action = (

            comparison

            .sort_values(
                "Expected Net Recovery",
                ascending=False
            )

            .iloc[0]["Action"]

        )


    selected = comparison[
        comparison["Action"]
        ==
        recommended_action
    ].iloc[0]


    return (
        recommended_action,
        comparison,
        selected
    )


# ============================================================
# FAST DASHBOARD OPTIMIZATION
# ============================================================
# The old version called optimize_action() for every transaction
# at startup. With 12,000+ rows this made Streamlit slow.
# We now calculate dashboard recommendations vectorially.
# Exact optimize_action() is still used for the selected transaction.

action_rates = (
    action_performance
    .set_index("action_taken")["historical_success_rate"]
    .to_dict()
)

for action_name in ACTION_COSTS:
    action_rates.setdefault(action_name, 0.30)

base_p = df["ml_recovery_probability"].to_numpy()
amounts = df["amount"].to_numpy()
fatigue = df["recovery_fatigue_level"].to_numpy()
hist_rate = df["historical_recovery_rate"].to_numpy()
reasons_arr = df["failure_reason"].to_numpy()
methods_arr = df["payment_method"].to_numpy()

action_scores = {}

for action_name, cost in ACTION_COSTS.items():

    p = (
        0.65 * base_p
        + 0.35 * action_rates[action_name]
    )

    # Failure-specific intelligence
    p += np.where(
        reasons_arr == "Temporary Bank/Network Error",
        0.12 if action_name == "Smart Retry" else
        (0.05 if action_name == "Delayed Retry" else 0),
        0
    )

    p += np.where(
        reasons_arr == "Insufficient Funds",
        0.12 if action_name == "Delayed Retry" else
        (0.06 if action_name == "Payment Reminder" else 0),
        0
    )

    p += np.where(
        reasons_arr == "Authentication Failed",
        0.12 if action_name == "Payment Link" else
        (-0.08 if action_name == "Smart Retry" else 0),
        0
    )

    p += np.where(
        reasons_arr == "Card Expired",
        0.18 if action_name == "Payment Method Update" else
        (-0.15 if action_name == "Smart Retry" else 0),
        0
    )

    # Payment-method signal
    p += np.where(
        (methods_arr == "UPI") & (action_name == "Smart Retry"),
        0.04,
        0
    )

    # Customer history
    p += np.where(
        hist_rate >= 0.60,
        0.05 if action_name in ["Smart Retry", "Delayed Retry"] else 0,
        0
    )

    p += np.where(
        hist_rate <= 0.20,
        0.08 if action_name == "Human Review" else 0,
        0
    )

    # Recovery fatigue
    p += np.where(
        fatigue == "MEDIUM",
        -0.05 if action_name == "Smart Retry" else 0,
        0
    )

    p += np.where(
        fatigue == "HIGH",
        -0.15 if action_name == "Smart Retry" else
        (-0.08 if action_name == "Delayed Retry" else
         (0.08 if action_name == "Human Review" else 0)),
        0
    )

    p += np.where(
        fatigue == "CRITICAL",
        -0.25 if action_name in ["Smart Retry", "Delayed Retry"] else
        (0.15 if action_name == "Human Review" else 0),
        0
    )

    p = np.clip(p, 0.01, 0.98)

    net = amounts * p - cost

    action_scores[action_name] = (p, net)

# Select the best action without iterating through 12,000 rows.
action_names = list(ACTION_COSTS.keys())
net_matrix = np.column_stack(
    [action_scores[a][1] for a in action_names]
)
best_idx = np.argmax(net_matrix, axis=1)

df["recommended_action"] = [
    action_names[i] for i in best_idx
]

df["optimized_probability"] = np.array([
    action_scores[action_names[i]][0][j]
    for j, i in enumerate(best_idx)
])

df["expected_recovery"] = (
    df["amount"] * df["optimized_probability"]
)

df["intervention_cost"] = df[
    "recommended_action"
].map(ACTION_COSTS)

df["expected_net_recovery"] = (
    df["expected_recovery"]
    - df["intervention_cost"]
)

# ============================================================
# FINAL DECISION
# ============================================================

def get_final_decision(row):

    if (
        row["recovery_fatigue_level"]
        ==
        "CRITICAL"
    ):

        return "HUMAN REVIEW"


    if row[
        "expected_net_recovery"
    ] <= 0:

        return "DO NOT PURSUE"


    return "RECOVER"


df["decision"] = df.apply(
    get_final_decision,
    axis=1
)


# ============================================================
# AI EXPLANATION
# ============================================================

def create_explanation(row):

    signals = []


    if row[
        "historical_recovery_rate"
    ] >= 0.50:

        signals.append(
            "strong customer recovery history"
        )

    elif row[
        "historical_recovery_rate"
    ] <= 0.20:

        signals.append(
            "low historical recovery rate"
        )


    if row[
        "previous_successful_payments"
    ] >= 8:

        signals.append(
            "strong previous payment behavior"
        )


    if row[
        "previous_recovery_success"
    ] == 1:

        signals.append(
            "previous recovery was successful"
        )


    if row[
        "failure_reason"
    ] == "Temporary Bank/Network Error":

        signals.append(
            "temporary failure favors retry"
        )


    if row[
        "failure_reason"
    ] == "Insufficient Funds":

        signals.append(
            "delayed recovery may be more effective"
        )


    if row[
        "failure_reason"
    ] == "Authentication Failed":

        signals.append(
            "customer action may be required"
        )


    if row[
        "failure_reason"
    ] == "Card Expired":

        signals.append(
            "payment method update is appropriate"
        )


    if row[
        "recovery_fatigue_level"
    ] in ["HIGH", "CRITICAL"]:

        signals.append(
            "multiple recovery attempts indicate fatigue"
        )


    if not signals:

        signals.append(
            "limited recovery signals available"
        )


    return (

        f"ML predicts a "
        f"{row['ml_recovery_probability']:.1%} "
        f"base recovery probability. "

        f"The customer's historical recovery rate is "
        f"{row['historical_recovery_rate']:.1%}. "

        f"Recovery fatigue is "
        f"{row['recovery_fatigue_level']} "
        f"({row['recovery_fatigue_score']}/100). "

        f"The optimizer recommends "
        f"{row['recommended_action']} "

        f"with expected net recovery of "
        f"₹{row['expected_net_recovery']:,.0f}. "

        f"Key signals: "
        + ", ".join(signals)
        + "."

    )


df["ai_explanation"] = df.apply(
    create_explanation,
    axis=1
)



# ============================================================
# AI-GENERATED PERSONALIZED RECOVERY MESSAGES
# ============================================================

def build_demo_recovery_message(row, channel, tone="Friendly & helpful", objective="Help fix the payment issue", extra=""):
    """Offline, deterministic recovery-message generator used when LLM access is unavailable."""
    amount = float(row["amount"])
    reason = str(row["failure_reason"])
    method = str(row["payment_method"])

    # Failure-specific action guidance. Keep it useful without inventing
    # links, discounts, deadlines, support numbers, or payment credentials.
    guidance = {
        "Insufficient Funds": (
            "Your payment could not be completed because the available balance "
            "may not have been sufficient. Please try again when funds are available."
        ),
        "Card Expired": (
            "Your saved card may have expired. Please update your payment method "
            "and try the payment again."
        ),
        "Authentication Failed": (
            "Your payment needs an authentication step to be completed. "
            "Please complete the authentication and try again."
        ),
        "Temporary Bank/Network Error": (
            "The payment could not be completed because of a temporary bank or "
            "network issue. Please try again shortly."
        ),
    }.get(
        reason,
        "Your payment could not be completed. Please check your payment details "
        "and try again."
    )

    if tone == "Professional":
        opener = "Hello,"
        closing = "Please try again when convenient."
    elif tone == "Gentle & empathetic":
        opener = "Hi there — we know payment issues can be frustrating."
        closing = "We hope this helps you complete your payment smoothly."
    elif tone == "Concise & action-oriented":
        opener = "Hi," 
        closing = "Please try again once the issue is resolved."
    else:
        opener = "Hi there,"
        closing = "Please try again when convenient."

    # Objective-aware emphasis.
    if objective == "Encourage a retry":
        action = "When you're ready, please try the payment again."
    elif objective == "Guide payment-method update":
        action = "Please update your payment method and try again."
    elif objective == "Reduce friction and complete payment":
        action = "Once the issue is resolved, you can try the payment again."
    else:
        action = guidance

    if objective == "Help fix the payment issue":
        body = guidance
    else:
        body = f"{guidance} {action}"

    if reason == "Card Expired":
        body = (
            "Your saved card may have expired. Please update your payment method "
            "and try the payment again."
        )
        if objective == "Encourage a retry":
            body += " Once updated, please try the payment again."

    if channel == "Email":
        return (
            f"Subject: Help completing your ₹{amount:,.0f} payment\n\n"
            f"{opener}\n\n{body}\n\n{closing}"
        )

    if channel == "WhatsApp":
        return f"{opener} Your ₹{amount:,.0f} payment needs attention. {body} {closing}"

    # SMS: concise and practical.
    return f"Hi, your ₹{amount:,.0f} payment needs attention. {body}"


def generate_personalized_recovery_message(row, channel, tone, objective, extra):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    context = f"""
Transaction amount: ₹{float(row['amount']):,.0f}
Payment method: {row['payment_method']}
Failure reason: {row['failure_reason']}
Customer segment: {row['customer_segment']}
Historical recovery rate: {float(row['historical_recovery_rate']):.1%}
Previous successful payments: {int(row['previous_successful_payments'])}
Previous recovery success: {"Yes" if int(row['previous_recovery_success']) == 1 else "No"}
Recovery attempts: {int(row['recovery_attempts'])}
Recovery fatigue: {row['recovery_fatigue_level']}
Recommended action: {row['recommended_action']}
Channel: {channel}
Tone: {tone}
Objective: {objective}
Extra instruction: {extra or "None"}
"""

    system = """
You are SidIQ, an AI revenue-recovery communication agent.
Write ONE natural, personalized payment-recovery message.

Use the supplied failure reason, amount, payment method and customer history.
Never reveal internal ML probabilities, customer segments, fatigue scores,
or internal decision logic.

Safety rules:
- Never request OTP, CVV, PIN, password, full card number or banking credentials.
- Never claim payment succeeded unless explicitly stated.
- Never invent discounts, deadlines, links, refunds, support numbers or policies.
- Never threaten, shame or pressure the customer.
- Insufficient funds: use a gentle, non-judgmental retry suggestion.
- Card expired: give actionable payment-method-update guidance.
- Authentication failed: suggest completing authentication and retrying.
- Temporary bank/network error: suggest trying again later.
- SMS should be concise.
- WhatsApp should be friendly and conversational.
- Email should contain a short subject and body.
Return only the message.
"""

    if OpenAI is not None and api_key:
        try:
            client = OpenAI(api_key=api_key)
            result = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.7,
                max_tokens=220,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
            )
            text = result.choices[0].message.content.strip()
            if text:
                return text, "LLM-generated"
        except Exception as exc:
            # OpenAI may return HTTP 429 when the API account has no credits or
            # has hit a usage/rate limit. Never expose the raw API error to the
            # customer-facing UI; switch silently to the guarded offline writer.
            status_code = getattr(exc, "status_code", None)
            message = str(exc).lower()
            is_429 = status_code == 429 or "429" in message or "rate limit" in message or "no credits" in message

            if is_429:
                return (
                    build_demo_recovery_message(row, channel, tone, objective, extra),
                    "Demo / Offline AI — API credits unavailable; using SidIQ guarded recovery writer"
                )

            return (
                build_demo_recovery_message(row, channel, tone, objective, extra),
                "Demo / Offline AI — LLM unavailable; using SidIQ guarded recovery writer"
            )

    return (
        build_demo_recovery_message(row, channel, tone, objective, extra),
        "Demo / Offline AI — no API key; using SidIQ guarded recovery writer"
    )


# ============================================================
# BUSINESS KPIs
# ============================================================

revenue_at_risk = df[
    "amount"
].sum()


total_expected_recovery = df[
    "expected_recovery"
].sum()


total_intervention_cost = df[
    "intervention_cost"
].sum()


total_net_recovery = (

    total_expected_recovery
    -
    total_intervention_cost

)


recoverable_cases = len(
    df[
        df["decision"]
        ==
        "RECOVER"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

st.divider()

st.subheader(
    "📊 Revenue Recovery Overview"
)


c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(
        "Revenue at Risk",
        f"₹{revenue_at_risk:,.0f}"
    )


with c2:

    st.metric(
        "Expected Recovery",
        f"₹{total_expected_recovery:,.0f}"
    )


with c3:

    st.metric(
        "Intervention Cost",
        f"₹{total_intervention_cost:,.0f}"
    )


with c4:

    st.metric(
        "Expected NET Recovery",
        f"₹{total_net_recovery:,.0f}"
    )


with c5:

    st.metric(
        "Recoverable Cases",
        recoverable_cases
    )


# ============================================================
# LEFT SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    '<div class="sidiq-sidebar-brand">\n<svg class="sidiq-sidebar-logo" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="SidIQ logo">\n  <defs>\n    <linearGradient id="sidBlue" x1="0" y1="0" x2="1" y2="1">\n      <stop offset="0%" stop-color="#1677FF"/>\n      <stop offset="100%" stop-color="#00B7FF"/>\n    </linearGradient>\n  </defs>\n  <path d="M24 4 42 14v20L24 44 6 34V14L24 4Z" fill="none" stroke="url(#sidBlue)" stroke-width="3.2" stroke-linejoin="round"/>\n  <path d="M13 31V22h6v9h-6Zm8 0V17h6v14h-6Zm8 0V13h6v18h-6Z" fill="url(#sidBlue)" opacity=".96"/>\n  <path d="M12 18l8-5 6 4 10-8" fill="none" stroke="#0B67D1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>\n  <path d="m33 9 3-.2-.8 3" fill="none" stroke="#0B67D1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>\n</svg>\n<div class="sidiq-sidebar-title">SidIQ</div></div>',
    unsafe_allow_html=True
)
st.sidebar.caption("Intelligent Revenue Recovery Platform")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "",
    [
        "📊 Dashboard",
        "🔎 AI Investigator",
        "🎯 Action Optimizer",
        "💬 AI Recovery Messages",
        "🧪 Recovery Lab",
        "🤖 ML Model",
        "📋 Transactions"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "CSV-based ML • Customer Memory • "
    "Recovery Fatigue • Action Optimization"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.subheader(
        "🎯 Recommended Action Distribution"
    )


    action_counts = (

        df[
            "recommended_action"
        ]

        .value_counts()

    )


    st.bar_chart(
        action_counts
    )


    st.subheader(
        "🥵 Recovery Fatigue Distribution"
    )


    fatigue_counts = (

        df[
            "recovery_fatigue_level"
        ]

        .value_counts()

    )


    st.bar_chart(
        fatigue_counts
    )


    st.subheader(
        "💰 Highest Net Recovery Opportunities"
    )


    priority_columns = [

        "transaction_id",

        "customer_id",

        "amount",

        "failure_reason",

        "ml_recovery_probability",

        "recommended_action",

        "expected_recovery",

        "intervention_cost",

        "expected_net_recovery",

        "recovery_fatigue_level",

        "decision"

    ]


    priority_df = (

        df

        .sort_values(
            "expected_net_recovery",
            ascending=False
        )

        [priority_columns]

        .head(20)

    )


    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI INVESTIGATOR
# ============================================================

elif page == "🔎 AI Investigator":

    st.subheader(
        "🔎 AI Transaction Investigator"
    )


    selected_transaction = st.selectbox(

        "Select a transaction",

        df["transaction_id"].tolist()

    )


    row = df[
        df["transaction_id"]
        ==
        selected_transaction
    ].iloc[0]


    left, right = st.columns(2)


    # --------------------------------------------------------
    # TRANSACTION INFORMATION
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### 💳 Transaction Information"
        )


        st.write(
            f"**Transaction ID:** "
            f"{row['transaction_id']}"
        )


        st.write(
            f"**Customer ID:** "
            f"{row['customer_id']}"
        )


        st.write(
            f"**Amount:** "
            f"₹{row['amount']:,.0f}"
        )


        st.write(
            f"**Payment Method:** "
            f"{row['payment_method']}"
        )


        st.write(
            f"**Failure Reason:** "
            f"{row['failure_reason']}"
        )


        st.write(
            f"**Previous Successful Payments:** "
            f"{int(row['previous_successful_payments'])}"
        )


        st.write(
            f"**Recovery Attempts:** "
            f"{int(row['recovery_attempts'])}"
        )


    # --------------------------------------------------------
    # CUSTOMER MEMORY
    # --------------------------------------------------------

    with left:

        st.divider()

        st.markdown(
            "### 🧠 Customer Recovery Memory"
        )


        m1, m2 = st.columns(2)


        with m1:

            st.metric(
                "Historical Recovery Rate",
                f"{row['historical_recovery_rate']:.1%}"
            )


        with m2:

            st.metric(
                "Past Recoveries",
                int(row["successful_recoveries"])
            )


        st.write(
            f"**Customer Segment:** "
            f"{row['customer_segment']}"
        )


        st.write(
            f"**Historical Transactions:** "
            f"{int(row['total_transactions'])}"
        )


        st.write(
            f"**Average Recovery Attempts:** "
            f"{row['average_recovery_attempts']:.1f}"
        )


    # --------------------------------------------------------
    # AI RESULT
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### 🤖 AI Decision"
        )


        st.metric(
            "ML Recovery Probability",
            f"{row['ml_recovery_probability']:.1%}"
        )


        st.metric(
            "Optimized Probability",
            f"{row['optimized_probability']:.1%}"
        )


        st.success(
            "🏆 Recommended Action: "
            +
            row["recommended_action"]
        )


        st.write(
            f"**Expected Gross Recovery:** "
            f"₹{row['expected_recovery']:,.0f}"
        )


        st.write(
            f"**Intervention Cost:** "
            f"₹{row['intervention_cost']:,.0f}"
        )


        st.write(
            f"**Expected NET Recovery:** "
            f"₹{row['expected_net_recovery']:,.0f}"
        )


        if row["decision"] == "RECOVER":

            st.success(
                "✅ Decision: RECOVER"
            )

        elif row["decision"] == "HUMAN REVIEW":

            st.warning(
                "👤 Decision: HUMAN REVIEW"
            )

        else:

            st.error(
                "⛔ Decision: DO NOT PURSUE"
            )


    # --------------------------------------------------------
    # FATIGUE
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 🥵 Recovery Fatigue Analysis"
    )


    f1, f2, f3 = st.columns(3)


    with f1:

        st.metric(
            "Fatigue Score",
            f"{row['recovery_fatigue_score']}/100"
        )


    with f2:

        st.metric(
            "Fatigue Level",
            row["recovery_fatigue_level"]
        )


    with f3:

        st.metric(
            "Recovery Attempts",
            int(row["recovery_attempts"])
        )


    if row[
        "recovery_fatigue_level"
    ] == "CRITICAL":

        st.error(
            "🚨 Critical recovery fatigue detected. "
            "Automated retries should stop and the case "
            "should be routed to human review."
        )

    elif row[
        "recovery_fatigue_level"
    ] == "HIGH":

        st.warning(
            "⚠️ High recovery fatigue detected. "
            "Aggressive retries should be avoided."
        )

    elif row[
        "recovery_fatigue_level"
    ] == "MEDIUM":

        st.info(
            "ℹ️ Moderate recovery fatigue detected."
        )

    else:

        st.success(
            "✅ Low recovery fatigue. Automated recovery "
            "is still appropriate."
        )


    # --------------------------------------------------------
    # EXPLAINABLE AI
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 🧠 Explainable AI"
    )


    st.info(
        row["ai_explanation"]
    )


# ============================================================
# ACTION OPTIMIZER
# ============================================================

elif page == "🎯 Action Optimizer":

    st.subheader(
        "🎯 AI Action Optimizer"
    )


    st.write(
        "SidIQ evaluates multiple recovery strategies "
        "using ML probability, historical action performance, "
        "customer behavior, recovery fatigue and intervention cost."
    )


    st.divider()


    optimizer_transaction = st.selectbox(

        "Choose a failed transaction",

        df["transaction_id"].tolist(),

        key="optimizer_transaction"

    )


    optimizer_row = df[
        df["transaction_id"]
        ==
        optimizer_transaction
    ].iloc[0]


    action, comparison, selected_action = optimize_action(
        optimizer_row
    )


    # --------------------------------------------------------
    # TRANSACTION SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Transaction Summary"
    )


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "Transaction Amount",
            f"₹{optimizer_row['amount']:,.0f}"
        )


    with s2:

        st.metric(
            "ML Probability",
            f"{optimizer_row['ml_recovery_probability']:.1%}"
        )


    with s3:

        st.metric(
            "Failure Reason",
            optimizer_row["failure_reason"]
        )


    with s4:

        st.metric(
            "Fatigue",
            optimizer_row["recovery_fatigue_level"]
        )


    # --------------------------------------------------------
    # CUSTOMER MEMORY
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Customer Memory"
    )


    cm1, cm2, cm3, cm4 = st.columns(4)


    with cm1:

        st.metric(
            "Historical Recovery Rate",
            f"{optimizer_row['historical_recovery_rate']:.1%}"
        )


    with cm2:

        st.metric(
            "Past Recoveries",
            int(optimizer_row["successful_recoveries"])
        )


    with cm3:

        st.metric(
            "Historical Transactions",
            int(optimizer_row["total_transactions"])
        )


    with cm4:

        st.metric(
            "Avg Recovery Attempts",
            f"{optimizer_row['average_recovery_attempts']:.1f}"
        )


    st.divider()


    # --------------------------------------------------------
    # STRATEGY COMPARISON
    # --------------------------------------------------------

    st.markdown(
        "### 🧮 Strategy Comparison"
    )


    display_comparison = comparison.copy()


    display_comparison[
        "Probability"
    ] = display_comparison[
        "Probability"
    ].apply(
        lambda x: f"{x:.1%}"
    )


    display_comparison[
        "Expected Recovery"
    ] = display_comparison[
        "Expected Recovery"
    ].apply(
        lambda x: f"₹{x:,.0f}"
    )


    display_comparison[
        "Cost"
    ] = display_comparison[
        "Cost"
    ].apply(
        lambda x: f"₹{x:,.0f}"
    )


    display_comparison[
        "Expected Net Recovery"
    ] = display_comparison[
        "Expected Net Recovery"
    ].apply(
        lambda x: f"₹{x:,.0f}"
    )


    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # AI RECOMMENDATION
    # --------------------------------------------------------

    st.markdown(
        "### 🏆 AI Recommendation"
    )


    if (
        optimizer_row[
            "recovery_fatigue_level"
        ]
        ==
        "CRITICAL"
    ):

        st.warning(
            "🚨 SidIQ recommends HUMAN REVIEW because "
            "the customer has reached critical recovery fatigue."
        )

    else:

        st.success(
            f"🏆 SidIQ recommends **{action}** because "
            f"it provides the highest expected net recovery "
            f"after considering ML probability, historical "
            f"action performance, customer history, fatigue "
            f"and intervention cost."
        )


    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            "Optimized Probability",
            f"{selected_action['Probability']:.1%}"
        )


    with r2:

        st.metric(
            "Expected Recovery",
            f"₹{selected_action['Expected Recovery']:,.0f}"
        )


    with r3:

        st.metric(
            "Expected NET Recovery",
            f"₹{selected_action['Expected Net Recovery']:,.0f}"
        )


    st.info(
        "NET Recovery = Expected Gross Recovery − Intervention Cost"
    )





# ============================================================
# AI RECOVERY MESSAGES
# ============================================================

elif page == "💬 AI Recovery Messages":

    st.subheader("💬 AI-Generated Personalized Recovery Messages")
    st.write(
        "SidIQ drafts channel-specific recovery communication using "
        "transaction context, failure reason and customer recovery history."
    )

    st.info(
        "🔐 Guardrails: the generator does not expose internal ML signals "
        "or request sensitive payment credentials."
    )

    st.divider()

    selected_message_txn = st.selectbox(
        "Select a failed transaction",
        df["transaction_id"].tolist(),
        key="message_transaction"
    )

    message_row = df[
        df["transaction_id"] == selected_message_txn
    ].iloc[0]

    left, right = st.columns(2)

    with left:
        st.markdown("### 👤 Customer & Payment Context")

        a, b = st.columns(2)
        with a:
            st.metric("Amount", f"₹{message_row['amount']:,.0f}")
            st.metric(
                "Historical Recovery",
                f"{message_row['historical_recovery_rate']:.1%}"
            )
        with b:
            st.metric("Failure", message_row["failure_reason"])
            st.metric("Fatigue", message_row["recovery_fatigue_level"])

        st.write(f"**Payment Method:** {message_row['payment_method']}")
        st.write(f"**Customer Segment:** {message_row['customer_segment']}")
        st.write(
            f"**Previous Successful Payments:** "
            f"{int(message_row['previous_successful_payments'])}"
        )
        st.write(
            f"**Recovery Attempts:** "
            f"{int(message_row['recovery_attempts'])}"
        )
        st.write(
            f"**SidIQ Recommended Action:** "
            f"**{message_row['recommended_action']}**"
        )

    with right:
        st.markdown("### ✍️ Message Controls")

        channel = st.selectbox(
            "Channel",
            ["WhatsApp", "SMS", "Email"],
            key="message_channel"
        )

        tone = st.selectbox(
            "Customer Tone",
            [
                "Friendly & helpful",
                "Professional",
                "Gentle & empathetic",
                "Concise & action-oriented"
            ],
            key="message_tone"
        )

        objective = st.selectbox(
            "Recovery Objective",
            [
                "Encourage a retry",
                "Help fix the payment issue",
                "Guide payment-method update",
                "Reduce friction and complete payment"
            ],
            key="message_objective"
        )

        extra = st.text_input(
            "Optional instruction",
            placeholder="e.g. Keep it short and reassuring",
            key="message_extra"
        )

        generate = st.button(
            "✨ Generate Personalized Message",
            type="primary",
            use_container_width=True,
            key="generate_message"
        )

    # Clear stale output whenever the message inputs change. This prevents an
    # Email draft from remaining visible after switching to WhatsApp/SMS, and
    # prevents a message for the previous transaction from being shown.
    message_input_signature = (
        str(message_row["transaction_id"]),
        str(channel),
        str(tone),
        str(objective),
        str(extra),
    )
    previous_signature = st.session_state.get("sidiq_message_input_signature")
    if previous_signature is not None and previous_signature != message_input_signature:
        st.session_state.pop("sidiq_generated_message", None)
        st.session_state.pop("sidiq_message_source", None)
    st.session_state["sidiq_message_input_signature"] = message_input_signature

    if generate:
        with st.spinner("SidIQ is generating the message..."):
            generated, source = generate_personalized_recovery_message(
                message_row,
                channel,
                tone,
                objective,
                extra
            )

        # Store the exact channel with the generated output so the UI can never
        # display a stale message under a newly selected channel.
        st.session_state["sidiq_generated_message"] = generated
        st.session_state["sidiq_message_source"] = source
        st.session_state["sidiq_message_generated_channel"] = channel
        st.session_state["sidiq_message_generated_transaction"] = str(message_row["transaction_id"])

    if (
        "sidiq_generated_message" in st.session_state
        and st.session_state.get("sidiq_message_generated_channel") == channel
        and st.session_state.get("sidiq_message_generated_transaction") == str(message_row["transaction_id"])
    ):

        st.divider()
        st.markdown("### 🤖 SidIQ Personalized Message")

        st.caption(
            st.session_state.get(
                "sidiq_message_source",
                "Generated"
            )
        )

        # IMPORTANT: use a unique widget key for each transaction/channel/input
        # combination. A fixed Streamlit widget key preserves its old value,
        # which can make a new WhatsApp/SMS draft display the previous Email
        # draft. The dynamic key prevents that stale-widget-state bug.
        import hashlib
        editable_key = "editable_sidiq_message_" + hashlib.md5(
            "|".join(message_input_signature).encode("utf-8")
        ).hexdigest()[:12]

        st.text_area(
            "Editable message",
            value=st.session_state["sidiq_generated_message"],
            height=180,
            key=editable_key
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("✓ Failure-aware")
        with c2:
            st.success("✓ Channel-aware")
        with c3:
            st.success("✓ Safety-guarded")

        st.markdown("### 🧠 Why SidIQ personalized it")

        reason = str(message_row["failure_reason"])

        if reason == "Insufficient Funds":
            st.info(
                "The agent uses a gentle tone because the customer may simply "
                "need to retry when funds are available."
            )
        elif reason == "Card Expired":
            st.info(
                "The agent gives an actionable payment-method update instead "
                "of repeatedly asking for a retry."
            )
        elif reason == "Authentication Failed":
            st.info(
                "The agent focuses on completing authentication before another attempt."
            )
        elif reason == "Temporary Bank/Network Error":
            st.info(
                "The agent uses a low-friction retry message because the failure "
                "may be temporary."
            )
        else:
            st.info(
                "The agent adapts the message to the available payment and "
                "customer-history signals."
            )

        st.markdown("### 🔄 Agentic Communication Loop")
        st.code(
            """
FAILED PAYMENT
      ↓
CUSTOMER + TRANSACTION CONTEXT
      ↓
RECOVERY DECISION
      ↓
PERSONALIZED LLM MESSAGE
      ↓
GUARDRAIL CHECK
      ↓
SEND VIA CHANNEL*
      ↓
PAYMENT OUTCOME
      ↓
LEARN FROM RESULT

* Prototype drafts the message; it does not actually send it.
""",
            language="text"
        )

# ============================================================
# RECOVERY LAB — 3 BUSINESS SIMULATION MODES
# ============================================================

elif page == "🧪 Recovery Lab":

    st.subheader("🧪 SidIQ Recovery Lab")
    st.write(
        "Explore recovery decisions before executing them. "
        "The Recovery Lab combines what-if analysis, customer-journey "
        "simulation, and budget optimization."
    )

    mode = st.radio(
        "Select Simulation Mode",
        [
            "🔬 What-If Simulator",
            "🧠 Customer Journey Simulator",
            "💰 Recovery Budget Optimizer"
        ],
        horizontal=True,
        key="recovery_lab_mode"
    )

    st.divider()

    # ========================================================
    # MODE 1 — WHAT-IF SIMULATOR
    # ========================================================
    if mode == "🔬 What-If Simulator":

        st.markdown("### 🔬 What-If Recovery Simulator")
        st.caption(
            "Change business assumptions and immediately estimate their "
            "impact on expected gross and NET recovery."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            wf_amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=100.0,
                max_value=10000000.0,
                value=15000.0,
                step=500.0,
                key="wf_amount"
            )

        with c2:
            wf_probability = st.slider(
                "Recovery Probability",
                1, 98, 60,
                1,
                format="%d%%",
                key="wf_probability"
            ) / 100

        with c3:
            wf_cost = st.number_input(
                "Intervention Cost (₹)",
                min_value=0.0,
                max_value=100000.0,
                value=5.0,
                step=1.0,
                key="wf_cost"
            )

        c4, c5 = st.columns(2)

        with c4:
            wf_retry_success = st.slider(
                "Retry Success Rate",
                1, 99, 72,
                1,
                format="%d%%",
                key="wf_retry_success"
            ) / 100

        with c5:
            wf_fatigue = st.select_slider(
                "Customer Fatigue",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                value="LOW",
                key="wf_fatigue"
            )

        # Fatigue adjustment.
        fatigue_adjustment = {
            "LOW": 0.00,
            "MEDIUM": -0.05,
            "HIGH": -0.15,
            "CRITICAL": -0.30
        }[wf_fatigue]

        adjusted_probability = float(
            np.clip(
                wf_probability
                + (wf_retry_success - 0.50) * 0.15
                + fatigue_adjustment,
                0.01,
                0.98
            )
        )

        expected_gross = wf_amount * adjusted_probability
        expected_net = expected_gross - wf_cost

        st.divider()
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Adjusted Probability", f"{adjusted_probability:.1%}")

        with m2:
            st.metric("Expected Recovery", f"₹{expected_gross:,.0f}")

        with m3:
            st.metric("Intervention Cost", f"₹{wf_cost:,.0f}")

        with m4:
            st.metric("Expected NET Recovery", f"₹{expected_net:,.0f}")

        st.markdown("### 📈 Sensitivity Analysis")

        scenarios = []
        for change in [-0.10, -0.05, 0.00, 0.05, 0.10]:
            scenario_probability = float(
                np.clip(adjusted_probability + change, 0.01, 0.98)
            )
            gross = wf_amount * scenario_probability
            net = gross - wf_cost

            scenarios.append({
                "Probability Change": f"{change:+.0%}",
                "Recovery Probability": f"{scenario_probability:.1%}",
                "Expected Recovery": f"₹{gross:,.0f}",
                "Expected NET Recovery": f"₹{net:,.0f}"
            })

        st.dataframe(
            pd.DataFrame(scenarios),
            use_container_width=True,
            hide_index=True
        )

        if expected_net > 0:
            st.success(
                "💡 Positive unit economics: the simulated intervention "
                "creates positive expected NET recovery."
            )
        else:
            st.warning(
                "⚠️ Negative unit economics: intervention cost exceeds "
                "the expected recovery value."
            )

    # ========================================================
    # MODE 2 — CUSTOMER JOURNEY SIMULATOR
    # ========================================================
    elif mode == "🧠 Customer Journey Simulator":

        st.markdown("### 🧠 Customer Journey Simulator")
        st.caption(
            "Simulate how SidIQ changes the recovery journey when repeated "
            "attempts begin creating customer fatigue."
        )

        j1, j2 = st.columns(2)

        with j1:
            journey_amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=100.0,
                max_value=10000000.0,
                value=18500.0,
                step=500.0,
                key="journey_amount"
            )

            journey_base_probability = st.slider(
                "Initial Recovery Probability",
                1, 98, 60,
                1,
                format="%d%%",
                key="journey_probability"
            ) / 100

        with j2:
            max_attempts = st.slider(
                "Maximum Automated Attempts",
                1, 6, 4,
                1,
                key="journey_attempts"
            )

            fatigue_threshold = st.slider(
                "Fatigue Threshold",
                2, 6, 3,
                1,
                key="journey_threshold"
            )

        actions = [
            "Payment Failed",
            "Smart Retry",
            "Delayed Retry",
            "Payment Reminder",
            "Payment Link",
            "Human Review"
        ]

        # Deterministic illustrative journey: the probability changes as
        # attempts accumulate and fatigue is detected.
        journey_rows = []
        probability = journey_base_probability
        recovered = False
        fatigue_detected = False

        for attempt in range(1, max_attempts + 1):

            if attempt == 1:
                action = "Smart Retry"
            elif attempt == 2:
                action = "Delayed Retry"
            else:
                action = "Payment Reminder"

            if attempt >= fatigue_threshold:
                fatigue_detected = True
                probability = max(0.05, probability - 0.15)

            # Use a fixed seed-like deterministic threshold so the demo
            # behaves consistently from run to run.
            outcome_score = ((attempt * 37) % 100) / 100
            success = outcome_score < probability

            if success:
                recovered = True
                journey_rows.append({
                    "Step": attempt,
                    "Action": action,
                    "Recovery Probability": f"{probability:.1%}",
                    "Status": "✅ RECOVERED"
                })
                break

            journey_rows.append({
                "Step": attempt,
                "Action": action,
                "Recovery Probability": f"{probability:.1%}",
                "Status": "❌ FAILED"
            })

        if not recovered and fatigue_detected:
            journey_rows.append({
                "Step": len(journey_rows) + 1,
                "Action": "Human Review",
                "Recovery Probability": "Re-evaluate",
                "Status": "🛑 AUTOMATION STOPPED"
            })

        st.divider()

        jmetrics = st.columns(4)

        with jmetrics[0]:
            st.metric(
                "Journey Status",
                "Recovered" if recovered else "Escalated"
            )

        with jmetrics[1]:
            st.metric(
                "Attempts Used",
                len(journey_rows)
            )

        with jmetrics[2]:
            st.metric(
                "Fatigue",
                "Detected" if fatigue_detected else "Not Detected"
            )

        with jmetrics[3]:
            st.metric(
                "Transaction",
                f"₹{journey_amount:,.0f}"
            )

        st.markdown("### 🔄 Simulated Recovery Journey")

        for row in journey_rows:
            st.write(
                f"**Step {row['Step']} → {row['Action']}**  |  "
                f"Probability: {row['Recovery Probability']}  |  "
                f"{row['Status']}"
            )

        if fatigue_detected and not recovered:
            st.warning(
                "🚨 SidIQ detected recovery fatigue and stopped automated "
                "attempts before escalating to Human Review."
            )
        elif recovered:
            st.success(
                "✅ Payment recovered during the simulated journey."
            )
        else:
            st.info(
                "ℹ️ No recovery was simulated within the selected journey."
            )

    # ========================================================
    # MODE 3 — RECOVERY BUDGET OPTIMIZER
    # ========================================================
    else:

        st.markdown("### 💰 Recovery Budget Optimizer")
        st.caption(
            "Give SidIQ a limited intervention budget. It ranks failed "
            "transactions by expected NET recovery and selects the best "
            "portfolio."
        )

        b1, b2, b3 = st.columns(3)

        with b1:
            budget = st.number_input(
                "Available Budget (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=50000.0,
                step=5000.0,
                key="budget"
            )

        with b2:
            max_cases = st.number_input(
                "Maximum Cases",
                min_value=10,
                max_value=50000,
                value=min(5000, len(df)),
                step=100,
                key="budget_cases"
            )

        with b3:
            budget_action = st.selectbox(
                "Intervention Strategy",
                list(ACTION_COSTS.keys()),
                key="budget_action"
            )

        if st.button(
            "🚀 Optimize Recovery Budget",
            type="primary",
            use_container_width=True,
            key="optimize_budget"
        ):

            budget_df = df.copy()

            # Use the trained model when available.
            if "recovery_probability" in budget_df.columns:
                budget_df["probability"] = pd.to_numeric(
                    budget_df["recovery_probability"],
                    errors="coerce"
                ).fillna(0.50)
            else:
                budget_df["probability"] = 0.50

            budget_df["probability"] = budget_df["probability"].clip(0.01, 0.98)

            budget_df["intervention_cost"] = float(
                ACTION_COSTS[budget_action]
            )

            budget_df["expected_recovery"] = (
                budget_df["amount"] * budget_df["probability"]
            )

            budget_df["expected_net_recovery"] = (
                budget_df["expected_recovery"]
                - budget_df["intervention_cost"]
            )

            # Rank by expected NET recovery per rupee spent.
            budget_df["net_per_rupee"] = (
                budget_df["expected_net_recovery"]
                / budget_df["intervention_cost"].replace(0, np.nan)
            )

            budget_df = budget_df.replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

            budget_df = budget_df.sort_values(
                "net_per_rupee",
                ascending=False
            ).head(int(max_cases))

            # Greedy budget allocation.
            budget_df["cumulative_cost"] = (
                budget_df["intervention_cost"].cumsum()
            )

            selected = budget_df[
                budget_df["cumulative_cost"] <= budget
            ].copy()

            selected_count = len(selected)
            used_budget = float(selected["intervention_cost"].sum())
            expected_recovery_total = float(
                selected["expected_recovery"].sum()
            )
            expected_net_total = float(
                selected["expected_net_recovery"].sum()
            )
            remaining_budget = max(0.0, budget - used_budget)

            st.divider()
            bm1, bm2, bm3, bm4 = st.columns(4)

            with bm1:
                st.metric(
                    "Cases Selected",
                    f"{selected_count:,}"
                )

            with bm2:
                st.metric(
                    "Budget Used",
                    f"₹{used_budget:,.0f}"
                )

            with bm3:
                st.metric(
                    "Expected Recovery",
                    f"₹{expected_recovery_total:,.0f}"
                )

            with bm4:
                st.metric(
                    "Expected NET Recovery",
                    f"₹{expected_net_total:,.0f}"
                )

            st.info(
                f"Remaining budget: ₹{remaining_budget:,.0f} | "
                f"Strategy: {budget_action}"
            )

            if selected_count == 0:
                st.warning(
                    "No transaction could be selected within the available "
                    "budget."
                )
            else:
                display_cols = [
                    c for c in [
                        "transaction_id",
                        "amount",
                        "probability",
                        "expected_recovery",
                        "intervention_cost",
                        "expected_net_recovery"
                    ] if c in selected.columns
                ]

                display_df = selected[display_cols].copy()

                if "amount" in display_df.columns:
                    display_df["amount"] = display_df["amount"].map(
                        lambda x: f"₹{x:,.0f}"
                    )

                if "probability" in display_df.columns:
                    display_df["probability"] = display_df["probability"].map(
                        lambda x: f"{x:.1%}"
                    )

                for col in [
                    "expected_recovery",
                    "intervention_cost",
                    "expected_net_recovery"
                ]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(
                            lambda x: f"₹{x:,.0f}"
                        )

                display_df.columns = [
                    c.replace("_", " ").title()
                    for c in display_df.columns
                ]

                st.markdown("### 🏆 Top Allocated Transactions")
                st.dataframe(
                    display_df.head(25),
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    f"SidIQ allocated the budget to {selected_count:,} "
                    f"high-value opportunities based on expected NET recovery."
                )


# ============================================================
# ML MODEL TAB
# ============================================================

elif page == "🤖 ML Model":

    st.subheader(
        "🤖 Machine Learning Model"
    )


    m1, m2, m3, m4, m5 = st.columns(5)


    with m1:

        st.metric(
            "Accuracy",
            f"{accuracy:.1%}"
        )


    with m2:

        st.metric(
            "Precision",
            f"{precision:.1%}"
        )


    with m3:

        st.metric(
            "Recall",
            f"{recall:.1%}"
        )


    with m4:

        st.metric(
            "F1 Score",
            f"{f1:.1%}"
        )


    with m5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.1%}"
        )


    st.divider()


    st.markdown(
        """
### 📂 Dataset Used

The ML model is trained using **transactions.csv**.

The model considers:

- Transaction amount
- Payment method
- Failure reason
- Previous successful payments
- Previous recovery success
- Recovery attempts
- Customer annual value
- Historical customer recovery rate
- Average historical recovery attempts

### 🎯 Action Optimization Dataset

The optimizer also uses **recovery_actions.csv**.

It learns the historical recovery success rate of:

- Smart Retry
- Delayed Retry
- Payment Reminder
- Payment Link
- Payment Method Update
- Human Review
"""
    )


    st.divider()


    st.markdown(
        "### 🔄 SidIQ AI Pipeline"
    )


    st.code(
        """
transactions.csv
       ↓
Data Processing
       ↓
Random Forest ML Model
       ↓
ML Recovery Probability
       ↓
Customer Recovery Memory
       ↓
Recovery Fatigue Detection
       ↓
recovery_actions.csv
       ↓
Historical Action Performance
       ↓
Action Optimizer
       ↓
Expected Gross Recovery
       ↓
Intervention Cost
       ↓
Expected NET Recovery
       ↓
🏆 Best Recovery Action
       ↓
RECOVER / HUMAN REVIEW / DO NOT PURSUE
        """,
        language="text"
    )


# ============================================================
# TRANSACTION DATA TAB
# ============================================================

elif page == "📋 Transactions":

    st.subheader(
        "📋 Transaction Dataset"
    )


    st.write(
        f"Loaded **{len(df):,} transactions** from "
        "`transactions.csv`."
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    csv_output = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        "📥 Download AI Results CSV",

        csv_output,

        "recoveriq_results.csv",

        "text/csv"

    )


    st.divider()


    st.subheader(
        "📊 Historical Recovery Action Data"
    )


    st.write(
        f"Loaded **{len(actions_df):,} historical recovery "
        "actions from `recovery_actions.csv`."
    )


    st.dataframe(
        actions_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SidIQ Prototype | CSV-Based ML Recovery Prediction | "
    "Customer Recovery Memory | Recovery Fatigue | "
    "Historical Action Learning | AI Action Optimization | "
    "Expected Net Recovery"
)