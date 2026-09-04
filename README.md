# 🚀 SidIQ — AI Revenue Recovery & Decision Intelligence Platform

SidIQ is an AI-powered revenue recovery platform designed to help businesses identify failed-payment revenue at risk, predict recovery probability, select the best recovery action, and optimize recovery decisions for maximum expected NET recovery.

The platform combines machine learning, customer recovery history, recovery fatigue, scenario simulation, and budget optimization into a single decision-intelligence workflow.

---

## 🎯 Problem Statement

Failed payments can result in significant revenue leakage.

A simple retry is not always the best solution because every customer and transaction is different.

SidIQ answers four key questions:

- Which failed transactions are most likely to be recovered?
- Why did the payment fail?
- What recovery action should be taken?
- Where should limited recovery resources be allocated to maximize NET recovery?

---

## 💡 Solution

SidIQ creates an intelligent recovery workflow:

**Detect → Predict → Decide → Simulate → Optimize**

The platform analyzes failed transactions and uses machine learning and business rules to recommend recovery strategies while considering intervention cost and customer fatigue.

---

## 🧠 Key Features

### 1. AI Investigator

Analyzes failed transactions and identifies important factors such as:

- Payment method
- Failure reason
- Transaction amount
- Previous successful payments
- Previous recovery history
- Recovery attempts
- Customer annual value

It helps explain **why a transaction may or may not be recoverable**.

---

### 2. ML Recovery Prediction

A machine learning model predicts the probability that a failed transaction can be recovered.

The model evaluation dashboard displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The model is trained and evaluated on the project's synthetic transaction dataset.

---

### 3. Customer Recovery Memory

SidIQ considers a customer's previous recovery behavior.

Historical recovery information is used to understand whether a customer has previously responded successfully to recovery interventions.

---

### 4. Recovery Fatigue

Repeated recovery attempts can reduce the effectiveness of further interventions.

SidIQ assigns a recovery fatigue score based on previous attempts and uses it when making recovery decisions.

This helps avoid excessive customer intervention.

---

### 5. AI Action Optimizer

The Action Optimizer evaluates available recovery actions and estimates their expected outcome.

It considers:

- Recovery probability
- Transaction value
- Intervention cost
- Customer fatigue

The objective is to maximize:

**Expected NET Recovery = Expected Gross Recovery − Intervention Cost**

---

# 🧪 Recovery Lab

Recovery Lab is the decision-simulation layer of SidIQ.

It contains three modes.

## 1. What-If Simulator

Tests different recovery scenarios by changing variables such as:

- Recovery probability
- Retry success rate
- Customer fatigue
- Intervention cost
- Transaction amount

The simulator shows how these changes affect expected recovery and NET recovery.

---

## 2. Customer Journey Simulator

Simulates the recovery journey of a failed transaction.

Example journey:

**Payment Failed → Smart Retry → Delayed Retry → Payment Reminder → Human Review**

The simulator tracks:

- Recovery attempts
- Customer fatigue
- Recovery probability
- Transaction status

This helps understand how recovery strategies evolve over multiple interventions.

---

## 3. Recovery Budget Optimizer

Businesses have limited resources for recovery interventions.

The Budget Optimizer allocates a fixed recovery budget across failed transactions.

It prioritizes transactions based on their expected NET recovery and recovery efficiency.

The goal is:

**Maximum Expected NET Recovery within a Limited Budget**

---

## 📊 Dashboard

The main dashboard provides a high-level view of:

- Revenue at Risk
- Expected Recovery
- Intervention Cost
- Expected NET Recovery
- Recoverable Cases
- ML Model Performance

---

## 🏗️ System Workflow

```text
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
