import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

# ------------------------ 1. Entropy Utility ------------------------

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()

# ---------------------- 2. Adaptation Function ----------------------

def adapt_model(model, test_data, entropy_weight, log_q_theta=0.0,
                device='cuda', n_adapt_steps=50, lr=1e-3, batch_size=64):
    """
    Adapt a single model using entropy minimization on test data.

    Args:
        model: A PyTorch model
        test_data: TensorDataset (test inputs only)
        entropy_weight: Coefficient for entropy term (α̃)
        log_q_theta: Log prior term (optional; default 0)
        device: Device for computation
        n_adapt_steps: Number of optimization steps
        lr: Learning rate
        batch_size: Batch size for test loader

    Returns:
        adapted_model: Entropy-adapted model
    """
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = deepcopy(model).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_adapt_steps):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            entropy = compute_entropy(logits)
            loss = entropy_weight * entropy - log_q_theta
            loss.backward()
            optimizer.step()

    return model

# ---------------------- 3. Inference Function -----------------------

def predict(model, test_data, device='cuda', batch_size=64):
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            preds.append(probs)
    return torch.cat(preds, dim=0)  # (m, num_classes)

# ---------------------- 4. Full BACS Pipeline -----------------------

def BACS_ensemble(base_model, test_inputs, entropy_weight, k=5,
                  device='cuda', n_adapt_steps=2, lr=1e-3, batch_size=64):
    """
    Full BACS adaptation using model ensemble and entropy minimization.

    Args:
        base_model: Pretrained model to clone
        test_inputs: Tensor of shape (m, ...)
        entropy_weight: Scalar α̃ for entropy weight
        k: Ensemble size
        device: CUDA/CPU
        n_adapt_steps: Number of test-time adaptation steps
        lr: Learning rate
        batch_size: Batch size for adaptation/inference

    Returns:
        final_probs: (m, num_classes) - ensemble-marginalized prediction
    """
    test_data = TensorDataset(test_inputs)

    # Step 1: Create and adapt each ensemble member
    adapted_members = []
    for i in range(k):
        # If using per-model prior: replace 0.0 with real log q_i(θ)
        adapted_model = adapt_model(base_model, test_data, entropy_weight,
                                    log_q_theta=0.0,
                                    device=device, n_adapt_steps=n_adapt_steps,
                                    lr=lr, batch_size=batch_size)
        adapted_members.append(adapted_model)

    # Step 2: Predict with each adapted model and average
    all_probs = []
    for model in adapted_members:
        probs = predict(model, test_data, device=device, batch_size=batch_size)
        all_probs.append(probs)

    final_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (m, num_classes)
    return final_probs
