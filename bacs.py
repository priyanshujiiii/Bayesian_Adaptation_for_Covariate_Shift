import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

# -------------------------------- 1. Utility --------------------------------

# 1.1 Entropy computation for softmax predictions
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()

# ----------------------- 2. Single-Model BACS Adaptation -----------------------

def BACS_single_model(model, test_data, entropy_weight, log_q_theta=0.0,
                      device='cuda', n_adapt_steps=10, lr=1e-3, batch_size=64):
    """
    BACS-style test-time adaptation using a single model.

    Args:
        model: Pretrained PyTorch model
        test_data: Tensor of shape (m, ...) - test inputs
        entropy_weight: Scalar α̃ for weighting entropy
        log_q_theta: Optional prior term (e.g., log q(θ)); set to 0 if not used
        device: 'cuda' or 'cpu'
        n_adapt_steps: Gradient steps to adapt on test data
        lr: Learning rate for optimizer
        batch_size: Test batch size

    Returns:
        adapted_probs: Tensor (m, num_classes) - prediction probabilities
    """
    
    # 2.1 Prepare test loader and model
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = deepcopy(model).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 2.2 Adapt model using test-time entropy loss
    for _ in range(n_adapt_steps):
        for x in test_loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            entropy = compute_entropy(logits)
            loss = entropy_weight * entropy - log_q_theta  # Maximize => minimize negative
            loss.backward()
            optimizer.step()

    # 2.3 Inference with adapted model
    model.eval()
    preds = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            preds.append(probs)

    adapted_probs = torch.cat(preds, dim=0)  # (m, num_classes)
    return adapted_probs
