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

## 💻 Project Structure

---

## 🚀 How to Run

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/bayesian-test-time-adaptation.git
cd bayesian-test-time-adaptation

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Run the experiment
python main.py --config configs/default.yaml
```
}


