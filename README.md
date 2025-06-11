# Training on Test Data with Bayesian Adaptation for Covariate Shift

**Reimplementation of NeurIPS 2022 paper by Aurick Zhou & Sergey Levine**  
*UC Berkeley, Department of Electrical Engineering and Computer Sciences*

---

## 📌 Project Summary

This project reimplements the key ideas from the paper:

> **Training on Test Data with Bayesian Adaptation for Covariate Shift**  
> [NeurIPS 2021](https://arxiv.org/pdf/2109.12746) — *Aurick Zhou, Sergey Levine*

The goal is to improve model performance under **distribution shifts** at test time, using a **Bayesian adaptation** framework. Instead of training a model to handle all possible shifts, this approach adapts to the specific test-time data using **unlabeled inputs** via **entropy minimization**.

---

## 🧠 Core Idea

- At test time, the model sees inputs from a different distribution.
- The method uses **Bayesian reasoning** to adapt the model to the test data without needing labels.
- This is achieved through a **regularized entropy minimization** process.
- Leads to better **accuracy** and **uncertainty estimation** on shifted data.

---


## 📌 Project Summary

This project reimplements the key ideas from the paper:

> **Training on Test Data with Bayesian Adaptation for Covariate Shift**  
> [NeurIPS 2021](https://arxiv.org/pdf/2109.12746) — *Aurick Zhou, Sergey Levine*

The goal is to improve model performance under **distribution shifts** at test time, using a **Bayesian adaptation** framework. Instead of training a model to handle all possible shifts, this approach adapts to the specific test-time data using **unlabeled inputs** via **entropy minimization**.

---

## 🧠 Core Idea

- At test time, the model sees inputs from a different distribution.
- The method uses **Bayesian reasoning** to adapt the model to the test data without needing labels.
- This is achieved through a **regularized entropy minimization** process.
- Leads to better **accuracy** and **uncertainty estimation** on shifted data.

---

## 🚀 Update & Experimental Notes

- This project successfully **replicates the core idea** of the BACS (Bayesian Adaptation with Calibration and Smoothing) algorithm.
- We observe that **increasing the number of BACS updates** significantly improves test-time performance, especially under heavy corruption.
- All experiments were run on **Google Colab** due to computational limitations. As a result, we did **not perform a full-scale replication** of the original results.
- Despite this, the core mechanism works well, and our results demonstrate **strong performance gains** under covariate shift scenarios.

---

## 🙏 Acknowledgment

We sincerely thank the authors for introducing this elegant and practical method.  
Their work inspired this reimplementation and enabled our exploration of robust test-time adaptation.

---
