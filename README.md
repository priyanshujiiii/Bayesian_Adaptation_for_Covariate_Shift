# Training on Test Data with Bayesian Adaptation for Covariate Shift

**Reimplementation of the NeurIPS Paper by Aurick Zhou & Sergey Levine**  
*Department of Electrical Engineering and Computer Sciences, UC Berkeley*

## 📌 Overview

This repository contains a reimplementation of the NeurIPS paper:

> **Training on Test Data with Bayesian Adaptation for Covariate Shift**  
> *Aurick Zhou, Sergey Levine*  
> NeurIPS 2022

The paper addresses the challenge of test-time covariate shift in deep learning. Instead of robustifying models against all possible shifts, the authors propose adapting models **at test time using unlabeled inputs** via a Bayesian formulation. This method improves both prediction **accuracy** and **uncertainty estimates** under distributional shift.

---

## 🧠 Abstract

> When faced with distribution shift at test time, deep neural networks often make inaccurate predictions with unreliable uncertainty estimates. While improving the robustness of neural networks is one promising approach to mitigate this issue, an appealing alternative is to directly adapt them to unlabeled inputs from the particular distribution shift encountered at test time.  
> However, this poses a challenging question: in the standard Bayesian model for supervised learning, unlabeled inputs are conditionally independent of model parameters when the labels are unobserved—so what can unlabeled data tell us about the model parameters at test-time?  
> In this paper, we derive a Bayesian model that provides a well-defined relationship between unlabeled inputs under distributional shift and model parameters, and show how approximate inference in this model can be instantiated with a simple regularized entropy minimization procedure at test-time. We evaluate our method on a variety of distribution shifts for image classification, including image corruptions, natural distribution shifts, and domain adaptation settings, and show that our method improves both accuracy and uncertainty estimation.

---

## ⚙️ Implementation Details

### ✅ Features
- Bayesian test-time adaptation to covariate shift
- Entropy minimization with regularization
- Evaluation on:
  - Image corruptions
  - Natural distribution shifts
  - Domain adaptation tasks

### 📦 Framework
- Implemented in **PyTorch** *(or update if different)*

### 📁 Project Structure
