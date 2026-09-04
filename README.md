# 🚀 SidIQ --- AI Revenue Recovery & Decision Intelligence Platform

SidIQ is an AI-powered revenue recovery platform designed to help
businesses identify failed-payment revenue at risk, predict recovery
probability, select the best recovery action, and optimize recovery
decisions for maximum expected NET recovery.

The platform combines machine learning, customer recovery history,
recovery fatigue, scenario simulation, budget optimization, and
personalized recovery communication into a single decision-intelligence
workflow.

------------------------------------------------------------------------

## 🎯 Problem Statement

Failed payments can result in significant revenue leakage.

A simple retry is not always the best solution because every customer
and transaction is different.

SidIQ answers four key questions:

-   Which failed transactions are most likely to be recovered?
-   Why did the payment fail?
-   What recovery action should be taken?
-   Where should limited recovery resources be allocated to maximize NET
    recovery?

------------------------------------------------------------------------

## 💡 Solution

SidIQ creates an intelligent recovery workflow:

**Detect → Predict → Decide → Simulate → Optimize → Personalize**

The platform analyzes failed transactions and uses machine learning and
business rules to recommend recovery strategies while considering
intervention cost, customer history, fatigue, and communication context.

------------------------------------------------------------------------

## 🧠 Key Features

### 1. AI Investigator

Analyzes failed transactions and identifies important factors such as:

-   Payment method
-   Failure reason
-   Transaction amount
-   Previous successful payments
-   Previous recovery history
-   Recovery attempts
-   Customer annual value

It helps explain **why a transaction may or may not be recoverable**.

------------------------------------------------------------------------

### 2. ML Recovery Prediction

A machine learning model predicts the probability that a failed
transaction can be recovered.

The model evaluation dashboard displays:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC

The model is trained and evaluated on the project's synthetic
transaction dataset.

> **Note:** Model metrics shown in the application are based on
> synthetic project data and should not be interpreted as production
> performance.

------------------------------------------------------------------------

### 3. Customer Recovery Memory

SidIQ considers a customer's previous recovery behavior.

Historical recovery information is used to understand whether a customer
has previously responded successfully to recovery interventions.

------------------------------------------------------------------------

### 4. Recovery Fatigue

Repeated recovery attempts can reduce the effectiveness of further
interventions.

SidIQ assigns a recovery fatigue score based on previous attempts and
uses it when making recovery decisions.

This helps avoid excessive customer intervention.

------------------------------------------------------------------------

### 5. AI Action Optimizer

The Action Optimizer evaluates available recovery actions and estimates
their expected outcome.

It considers:

-   Recovery probability
-   Transaction value
-   Intervention cost
-   Customer fatigue
-   Historical action performance

The objective is to maximize:

**Expected NET Recovery = Expected Gross Recovery − Intervention Cost**

------------------------------------------------------------------------

### 6. AI Personalized Recovery Messages

SidIQ can generate personalized recovery communication for failed
payments.

Instead of using the same generic reminder for every customer, the
message is adapted using:

-   Failure reason
-   Transaction amount
-   Customer recovery history
-   Customer segment
-   Recovery fatigue
-   Selected communication channel
-   Customer tone
-   Recovery objective

Supported channels include:

-   WhatsApp
-   SMS
-   Email

Examples of failure-aware personalization:

-   **Insufficient Funds:** gentle and reassuring retry guidance
-   **Card Expired:** actionable payment-method update instructions
-   **Payment Gateway Timeout:** concise guidance to retry later
-   **Payment Method Issue:** clear next-step instructions

The feature includes communication guardrails to avoid requesting
sensitive information such as OTPs, CVVs, PINs, passwords, or banking
credentials.

### Offline / Demo Fallback

If an OpenAI API call is unavailable or the API account has no remaining
credits, SidIQ automatically switches to a local guarded
recovery-message writer.

This keeps the feature functional during demos without requiring live
API credits.

The UI clearly identifies this as:

**Demo / Offline AI --- using SidIQ guarded recovery writer**

When an API is available, the platform can use an LLM to generate the
message dynamically.

> **Important:** The offline fallback is a deterministic local
> generator, not an LLM. It should be described as an offline/demo
> recovery writer when the API is unavailable.

------------------------------------------------------------------------

# 🧮 Recovery Decision Intelligence

SidIQ is designed to optimize **incremental revenue created by an
intervention**, rather than looking only at recovery probability.

The decision logic can be expressed as:

``` text
FAILED PAYMENT
       ↓
Recovery Probability
       ↓
Counterfactual Test
       ↓
"What if we did nothing?"
       ↓
Incremental Recovery
       ↓
Intervention Cost
       ↓
Incremental NET Recovery
       ↓
Guardrail Check
       ↓
ACT / STOP / HUMAN REVIEW
```

### Counterfactual Recovery Engine

A future decision layer can compare:

-   Expected recovery with an intervention
-   Expected recovery without an intervention
-   Intervention cost
-   Expected incremental NET recovery

Example:

``` text
No action expected recovery:       ₹2,100
Smart Retry expected recovery:     ₹6,500
Intervention cost:                    ₹80
Incremental recovery:              ₹4,400
Incremental NET recovery:          ₹4,320

Decision: PROCEED
```

This reframes recovery from:

**"Which action has the highest recovery probability?"**

to:

**"Which action creates the most incremental NET revenue compared with
doing nothing?"**

------------------------------------------------------------------------

# 🛑 Recovery Kill-Switch & Guardrails

A recovery system should not keep contacting customers simply because a
transaction remains unpaid.

SidIQ can apply bounded decision rules such as:

-   Stop when recovery fatigue becomes critical
-   Stop when expected incremental recovery is lower than intervention
    cost
-   Escalate uncertain or high-value cases to human review
-   Respect maximum automated recovery attempts
-   Avoid unnecessary repeated communication
-   Maintain an audit trail of recovery decisions

Conceptually:

``` text
Recovery Opportunity
        ↓
Check Probability
        ↓
Check Incremental NET Value
        ↓
Check Customer Fatigue
        ↓
Check Recovery Policy
        ↓
 ┌──────┼─────────┐
 ↓      ↓         ↓
 ACT   HUMAN     STOP
      REVIEW
```

------------------------------------------------------------------------

# 🧬 Recovery DNA

SidIQ can build a customer-level **Recovery DNA** profile from
historical behavior.

The profile can learn signals such as:

-   Retry responsiveness
-   Reminder responsiveness
-   Historical recovery rate
-   Recovery attempt tolerance
-   Preferred recovery action
-   Fatigue sensitivity

This allows recovery decisions to become increasingly customer-specific
instead of relying only on transaction-level information.

------------------------------------------------------------------------

# 📈 Recovery Strategy Evolution

Recovery performance can be measured continuously by comparing:

-   Action selected
-   Action cost
-   Recovery outcome
-   Customer segment
-   Failure reason
-   Historical action effectiveness

This creates a foundation for a self-learning recovery policy that can
improve action selection as more recovery outcomes are collected.

------------------------------------------------------------------------

# 🧪 Recovery Lab

Recovery Lab is the decision-simulation layer of SidIQ.

It contains three modes.

## 1. What-If Simulator

Tests different recovery scenarios by changing variables such as:

-   Recovery probability
-   Retry success rate
-   Customer fatigue
-   Intervention cost
-   Transaction amount

The simulator shows how these changes affect expected recovery and NET
recovery.

------------------------------------------------------------------------

## 2. Customer Journey Simulator

Simulates the recovery journey of a failed transaction.

Example journey:

**Payment Failed → Smart Retry → Delayed Retry → Payment Reminder →
Human Review**

The simulator tracks:

-   Recovery attempts
-   Customer fatigue
-   Recovery probability
-   Recovery transaction status

This helps understand how recovery strategies evolve over multiple
interventions.

------------------------------------------------------------------------

## 3. Recovery Budget Optimizer

Businesses have limited resources for recovery interventions.

The Budget Optimizer allocates a fixed recovery budget across failed
transactions.

It prioritizes transactions based on their expected NET recovery and
recovery efficiency.

The goal is:

**Maximum Expected NET Recovery within a Limited Budget**

------------------------------------------------------------------------

# 📊 Dashboard

The main dashboard provides a high-level view of:

-   Revenue at Risk
-   Expected Recovery
-   Intervention Cost
-   Expected NET Recovery
-   Recoverable Cases
-   ML Model Performance

------------------------------------------------------------------------

# 🏗️ System Workflow

``` text
Failed Transactions
        ↓
AI Investigator
        ↓
ML Recovery Prediction
        ↓
Customer Recovery Memory
        ↓
Recovery Fatigue Analysis
        ↓
AI Action Optimizer
        ↓
Recovery Lab
   ┌────┴───────────────┐
   ↓                    ↓
What-If            Customer Journey
Simulator             Simulator
   │
   └──────────┬─────────┘
              ↓
     Recovery Budget
        Optimizer
              ↓
    Expected NET Recovery
              ↓
  Personalized Recovery Message
              ↓
     Guardrail / Policy Check
              ↓
       Recovery Outcome
              ↓
       Learn from Result
```

------------------------------------------------------------------------

# 🛠️ Technology Stack

-   **Python**
-   **Streamlit**
-   **Pandas**
-   **NumPy**
-   **Scikit-learn**
-   **Plotly**
-   **OpenAI API (optional for live LLM message generation)**

------------------------------------------------------------------------

# 📂 Project Structure

``` text
SidIQ-AI-Revenue-Recovery/
│
├── app.py
├── transactions.csv
├── recovery_actions.csv
├── requirements.txt
├── README.md
│
└── outputs/
    └── README.md
```

------------------------------------------------------------------------

# 📊 Dataset

SidIQ currently uses synthetic transaction and recovery-action data for
demonstration.

### transactions.csv

Contains transaction-level information such as:

-   Transaction ID
-   Customer ID
-   Amount
-   Payment method
-   Failure reason
-   Previous successful payments
-   Previous recovery success
-   Recovery attempts
-   Customer annual value
-   Recovery outcome

### recovery_actions.csv

Contains historical recovery-action information such as:

-   Transaction ID
-   Customer ID
-   Action taken
-   Intervention cost
-   Recovery probability at action
-   Recovery outcome

------------------------------------------------------------------------

# 🚀 Run Locally

Clone the repository:

``` bash
git clone https://github.com/SiddhantArora091/SidIQ-AI-Revenue-Recovery.git
```

Enter the project directory:

``` bash
cd SidIQ-AI-Revenue-Recovery
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run Streamlit:

``` bash
python -m streamlit run app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## 🔑 Optional: Enable Live LLM Recovery Messages

The personalized-message feature works in offline/demo mode without API
credits.

For live LLM generation, set an OpenAI API key as an environment
variable.

### Windows PowerShell

``` powershell
$env:OPENAI_API_KEY="your_api_key_here"
python -m streamlit run app.py
```

Do **not** put your API key directly inside `app.py` or commit it to
GitHub.

------------------------------------------------------------------------

# 🔐 Safety & Guardrails

SidIQ's recovery communication design avoids:

-   OTP requests
-   CVV requests
-   PIN requests
-   Password requests
-   Full card details
-   Banking credentials
-   False payment-success claims
-   Invented discounts or deadlines
-   Unverified support links
-   Pressure, shame, or manipulative language

The prototype is designed around bounded recovery actions and
explainable decision logic.

------------------------------------------------------------------------

# 🔮 Future Scope

Potential extensions include:

-   Counterfactual Recovery Engine
-   Automated Recovery Kill-Switch
-   Customer Recovery DNA
-   Self-learning recovery strategy
-   Recovery opportunity-cost optimization
-   Intervention collision detection
-   Optimal recovery contact timing
-   Human-review capacity optimization
-   Model uncertainty and abstention
-   Portfolio-level recovery stress testing
-   Recovery audit trail and decision replay
-   Real WhatsApp/SMS/email provider integration
-   Production payment-system integration

------------------------------------------------------------------------

# 🏆 Project Positioning

SidIQ is not designed to simply predict whether a failed payment will
recover.

Its goal is to answer:

> **What should we do, why should we do it, how much should we spend,
> when should we stop, and how much incremental NET revenue did the
> intervention create?**

The platform combines **prediction + decisioning + simulation +
optimization + personalized communication** into one revenue-recovery
workflow.

------------------------------------------------------------------------

## ⚠️ Disclaimer

SidIQ is an internship/buildathon prototype.

The datasets are synthetic and the results are intended for
demonstration and experimentation only. Production deployment would
require real payment data, security controls, compliance validation,
model monitoring, communication-provider integration, and
merchant-defined recovery policies.
