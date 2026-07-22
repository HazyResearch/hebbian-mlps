"""Margin metrics for the associative-recall task.

These functions measure how well the attention output at Q matches the target
value embedding for ``<junk_prefix> K <junk_suffix> Q V`` samples.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from hebbian.transformer.model import GPT


def compute_values_margin_l2_distribution(
    sorted_value_matrix: torch.Tensor,
    num_samples: int = 5_000,
    max_epsilon_magnitude: float = 10.0,
    batch_size: int = 100,
    device: str | None = None,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Compute per-value maximum L2 epsilon such that decoding is correct.

    For each value i, finds the maximum epsilon magnitude such that:
    argmax_j (v_i + eps)^T v_j = i
    for all sampled perturbations eps of that magnitude.

    Uses parallel binary search across all values simultaneously.

    Args:
        sorted_value_matrix: (n, d) tensor of sorted value embeddings v_j
        num_samples: Number of perturbations to sample per magnitude per value
        max_epsilon_magnitude: Maximum magnitude to search
        batch_size: Batch size for processing perturbations
        device: Device to use
        tolerance: Tolerance for binary search

    Returns:
        NumPy array of per-value maximum L2 margins (epsilon magnitudes)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    V = sorted_value_matrix.to(device)
    n, d = V.shape
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    V_t = V.T  # (d, n)

    def test_values_at_magnitudes(magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Test each value at its corresponding epsilon magnitude.

        Args:
            magnitudes: (n,) tensor of magnitudes to test, one per value

        Returns:
            (n,) boolean tensor - True if value i passes all perturbations at magnitudes[i]
        """
        with torch.no_grad():
            key_all_correct = torch.ones(n, device=device, dtype=torch.bool)

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                current_batch_size = batch_end - batch_start

                # Sample random directions (unit vectors) - one per (perturbation, value)
                random_vectors = torch.randn(current_batch_size, n, d, device=device)
                directions = F.normalize(random_vectors, p=2, dim=-1)

                # Scale each value's perturbations by its own magnitude
                epsilons = directions * magnitudes.view(1, n, 1)  # (batch_size, n, d)

                # Add epsilon to each value: (batch_size, n, d)
                perturbed = V.unsqueeze(0) + epsilons

                # Compute scores: (batch_size, n, n) where [b, i, j] = (v_i + eps_b)^T v_j
                scores = torch.bmm(perturbed, V_t.unsqueeze(0).expand(current_batch_size, -1, -1))

                # Check if argmax_j (v_i + eps)^T v_j = i for each value i
                predicted = torch.argmax(scores, dim=-1)  # (batch_size, n)
                expected = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand_as(predicted)
                correct_per_key = (predicted == expected)  # (batch_size, n)

                batch_all_correct = correct_per_key.all(dim=0)  # (n,)
                key_all_correct &= batch_all_correct

            return key_all_correct

    # Parallel binary search across all values
    low = torch.zeros(n, device=device)
    high = torch.full((n,), float(max_epsilon_magnitude), device=device)
    best_valid = torch.zeros(n, device=device)

    while (high - low).max() > tolerance:
        mid = (low + high) / 2.0
        passes = test_values_at_magnitudes(mid)
        best_valid = torch.where(passes, mid, best_valid)
        low = torch.where(passes, mid, low)
        high = torch.where(passes, high, mid)

    return best_valid.detach().cpu().numpy()

def compute_values_margin_angular_distribution(
    sorted_value_matrix: torch.Tensor,
    num_samples: int = 5_000,
    max_epsilon_angle: float = np.pi,
    batch_size: int = 100,
    device: str | None = None,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    For each key v_i, compute the largest angle theta such that for all
    tangent directions u ⟂ v_i (empirically sampled),

        argmax_j (cos(theta) v_i + sin(theta) u)^T v_j = i.

    This measures a worst-case angular decoding margin.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    V = sorted_value_matrix.to(device)
    n, d = V.shape
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Normalize keys (pure angular decoding)
    V_hat = F.normalize(V, p=2, dim=-1)          # (n, d)
    V_hat_T = V_hat.T                            # (d, n)

    def test_values_at_angles(angles: torch.Tensor) -> torch.Tensor:
        """
        Test whether each key i survives angle[i] for all sampled
        tangent perturbations.
        """
        with torch.no_grad():
            cos_theta = torch.cos(angles)         # (n,)
            sin_theta = torch.sin(angles)         # (n,)

            key_all_correct = torch.ones(n, device=device, dtype=torch.bool)

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                B = batch_end - batch_start

                # Sample random tangent directions u ⟂ v_i
                random_vectors = torch.randn(B, n, d, device=device)
                proj = (random_vectors * V_hat.unsqueeze(0)).sum(dim=-1, keepdim=True)
                orthogonal = random_vectors - proj * V_hat.unsqueeze(0)
                orthogonal = F.normalize(orthogonal, dim=-1, eps=1e-12)

                # Construct exact angular perturbations
                perturbed = (
                    cos_theta.view(1, n, 1) * V_hat.unsqueeze(0)
                    + sin_theta.view(1, n, 1) * orthogonal
                )  # (B, n, d), already unit norm

                # Angular scores
                scores = torch.bmm(
                    perturbed,
                    V_hat_T.unsqueeze(0).expand(B, -1, -1),
                )  # (B, n, n)

                predicted = torch.argmax(scores, dim=-1)  # (B, n)
                expected = torch.arange(n, device=device).view(1, n).expand(B, n)

                key_all_correct &= (predicted == expected).all(dim=0)

                # Early exit if everything already failed
                if not key_all_correct.any():
                    break

            return key_all_correct

    # Parallel binary search for each key
    low = torch.zeros(n, device=device)
    high = torch.full((n,), min(float(max_epsilon_angle), float(np.pi)), device=device)
    best_valid = torch.zeros(n, device=device)

    while (high - low).max() > tolerance:
        mid = (low + high) / 2.0
        passes = test_values_at_angles(mid)

        best_valid = torch.where(passes, mid, best_valid)
        low = torch.where(passes, mid, low)
        high = torch.where(passes, high, mid)

    return best_valid.cpu().numpy()


def _get_num_facts(dataloader) -> int:
    """Extract num_facts from a batch generator or DataLoader."""
    if hasattr(dataloader, 'num_facts'):
        return dataloader.num_facts
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'num_facts'):
        return dataloader.dataset.num_facts
    raise ValueError("Cannot determine num_facts from dataloader")


def compute_attention_output_margin_l2_distribution(
    gpt_model: GPT,
    dataloader: DataLoader,
    device: str | None = None,
    num_batches: int = 100,
    use_ln_2: bool = False,
) -> np.ndarray:
    """
    Compute per-key L2 error between a layer norm output and the key embedding K.

    Measures error at Q for ``<junk_prefix> K <junk_suffix> Q V`` samples.

    At the prediction position the model output is compared to the key
    embedding ``wte[K]``.  The key token K is identified as the first token
    in ``[0, num_facts)`` in the input (junk and Q tokens are >= num_facts).

    Args:
        gpt_model: The GPT model (should be configured for attention-only training)
        dataloader: DataLoader or batch generator for AssociativeRecallDataset
        device: Device to use (defaults to cuda if available)
        num_batches: Maximum number of batches to process
        use_ln_2: If True, hook into ln_2 (layer norm before MLP) instead of
                  ln_f (final layer norm before lm_head)

    Returns:
        NumPy array of per-key maximum L2 errors
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gpt_model.eval()
    gpt_model.to(device)
    
    # Get input embeddings (wte weights)
    wte = gpt_model.transformer.wte.weight  # (vocab_size, d)
    num_facts = _get_num_facts(dataloader)
    
    # Track maximum error per key token
    key_max_errors = {}
    num_processed = 0
    
    # Choose which layer norm to hook
    hook_module = (
        gpt_model.transformer.h[0].ln_2 if use_ln_2
        else gpt_model.transformer.ln_f
    )
    
    # Use iterator that restarts if dataloader empties before num_batches
    dataloader_iter = iter(dataloader)
    while num_processed < num_batches:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if exhausted
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Hook to capture layer norm output
        ln_outputs = []
        
        def ln_hook(module, input, output):
            ln_outputs.append(output.detach())
        
        hook_handle = hook_module.register_forward_hook(ln_hook)
        
        with torch.no_grad():
            _ = gpt_model(inputs)
        
        hook_handle.remove()
        
        if not ln_outputs:
            num_processed += 1
            continue
        
        attn_out = ln_outputs[0]  # (B, T, C)
        batch_size = inputs.shape[0]
        
        for b in range(batch_size):
            # Find prediction position (where target != -100)
            pred_positions = torch.where(targets[b] != -100)[0]
            if len(pred_positions) == 0:
                continue
            pred_pos = pred_positions[0].item()
            
            # Find the key token K: first token in [0, num_facts) in the input
            key_mask = (inputs[b] >= 0) & (inputs[b] < num_facts)
            key_positions = torch.where(key_mask)[0]
            if len(key_positions) == 0:
                continue
            key_token = inputs[b, key_positions[0]].item()
            
            # Get model output at prediction position
            predicted = attn_out[b, pred_pos]  # (C,)
            
            # Get the key embedding (the target we compare against)
            intended = wte[key_token]  # (C,)
            
            # Compute L2 error
            error = torch.norm(predicted - intended, p=2).item()
            
            # Track maximum error per key
            if key_token not in key_max_errors:
                key_max_errors[key_token] = error
            else:
                key_max_errors[key_token] = max(key_max_errors[key_token], error)
        
        num_processed += 1
    
    if not key_max_errors:
        return np.zeros(0, dtype=np.float32)
    
    return np.array(list(key_max_errors.values()), dtype=np.float32)


def compute_attention_output_margin_angular_distribution(
    gpt_model: GPT,
    dataloader: DataLoader,
    device: str | None = None,
    num_batches: int = 100,
    use_ln_2: bool = False,
) -> np.ndarray:
    """
    Compute per-key angular error between a layer norm output and the key embedding K.

    Measures error at Q for ``<junk_prefix> K <junk_suffix> Q V`` samples.

    At the prediction position the model output is compared to the key
    embedding ``wte[K]``.  The key token K is identified as the first token
    in ``[0, num_facts)`` in the input (junk and Q tokens are >= num_facts).

    Args:
        gpt_model: The GPT model (should be configured for attention-only training)
        dataloader: DataLoader or batch generator for AssociativeRecallDataset
        device: Device to use (defaults to cuda if available)
        num_batches: Maximum number of batches to process
        use_ln_2: If True, hook into ln_2 (layer norm before MLP) instead of
                  ln_f (final layer norm before lm_head)

    Returns:
        NumPy array of per-key maximum angular errors (in radians)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gpt_model.eval()
    gpt_model.to(device)
    
    # Get input embeddings (wte weights)
    wte = gpt_model.transformer.wte.weight  # (vocab_size, d)
    num_facts = _get_num_facts(dataloader)
    
    # Track maximum angle per key token
    key_max_angles = {}
    num_processed = 0
    
    # Choose which layer norm to hook
    hook_module = (
        gpt_model.transformer.h[0].ln_2 if use_ln_2
        else gpt_model.transformer.ln_f
    )
    
    # Use iterator that restarts if dataloader empties before num_batches
    dataloader_iter = iter(dataloader)
    while num_processed < num_batches:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if exhausted
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Hook to capture layer norm output
        ln_outputs = []
        
        def ln_hook(module, input, output):
            ln_outputs.append(output.detach())
        
        hook_handle = hook_module.register_forward_hook(ln_hook)
        
        with torch.no_grad():
            _ = gpt_model(inputs)
        
        hook_handle.remove()
        
        if not ln_outputs:
            num_processed += 1
            continue
        
        attn_out = ln_outputs[0]  # (B, T, C)
        batch_size = inputs.shape[0]
        
        for b in range(batch_size):
            # Find prediction position (where target != -100)
            pred_positions = torch.where(targets[b] != -100)[0]
            if len(pred_positions) == 0:
                continue
            pred_pos = pred_positions[0].item()
            
            # Find the key token K: first token in [0, num_facts) in the input
            key_mask = (inputs[b] >= 0) & (inputs[b] < num_facts)
            key_positions = torch.where(key_mask)[0]
            if len(key_positions) == 0:
                continue
            key_token = inputs[b, key_positions[0]].item()
            
            # Get model output at prediction position
            predicted = attn_out[b, pred_pos]  # (C,)
            
            # Get the key embedding (the target we compare against)
            intended = wte[key_token]  # (C,)
            
            # Normalize both vectors
            predicted_norm = F.normalize(predicted, p=2, dim=0, eps=1e-12)
            intended_norm = F.normalize(intended, p=2, dim=0, eps=1e-12)
            
            # Compute cosine similarity (dot product of normalized vectors)
            cos_angle = torch.clamp(
                torch.dot(predicted_norm, intended_norm),
                -1.0, 1.0
            )
            
            # Compute angular error in radians
            angle = torch.arccos(cos_angle).item()
            
            # Track maximum angle per key
            if key_token not in key_max_angles:
                key_max_angles[key_token] = angle
            else:
                key_max_angles[key_token] = max(key_max_angles[key_token], angle)
        
        num_processed += 1
    
    if not key_max_angles:
        return np.zeros(0, dtype=np.float32)
    
    return np.array(list(key_max_angles.values()), dtype=np.float32)


def compute_mixer_input_margin_angular_distr(
    model: torch.nn.Module,
    input_keys: torch.Tensor,
    sorted_value_matrix: torch.Tensor,
    num_samples: int = 5_000,
    max_epsilon_angle: float = np.pi,
    batch_size: int = 100,
    device: str | None = None,
    tolerance: float = 1e-4,
    noise_subspace: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Compute the per-key maximum angle such that decoding is correct, and return the
    full distribution of these per-key maximum angles.

    This mirrors `compute_mixer_input_margin_angular`, but instead of returning
    quantiles of the per-key maximum angles, it returns the raw distribution as
    a NumPy array of shape (n_keys,).

    Args:
        noise_subspace: Optional (m, d) tensor of vectors spanning the subspace
            to which noise is restricted. If None, noise is sampled from all of R^d.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    K = input_keys.to(device)
    V = sorted_value_matrix.to(device)
    n, d = K.shape

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    K_hat = F.normalize(K, p=2, dim=-1)
    V_t = V.T  # (d, n)

    # Precompute orthonormal basis for noise subspace if provided
    if noise_subspace is not None:
        # noise_subspace: (m, d) -> Q: (d, r) where r = rank
        Q, _ = torch.linalg.qr(noise_subspace.T.to(device))
        subspace_dim = Q.shape[1]
    else:
        Q = None
        subspace_dim = d

    def test_keys_at_angles(angles: torch.Tensor) -> torch.Tensor:
        """
        Test each key at its corresponding angle.

        Args:
            angles: (n,) tensor of angles to test, one per key

        Returns:
            (n,) boolean tensor - True if key i passes all perturbations at angles[i]
        """
        with torch.no_grad():
            cos_theta = torch.cos(angles)  # (n,)
            sin_theta = torch.sin(angles)  # (n,)

            # Track correct predictions per key across all perturbations
            key_all_correct = torch.ones(n, device=device, dtype=torch.bool)

            # Test all perturbations in batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                current_batch_size = batch_end - batch_start

                # Sample random directions and make them orthogonal to k_i
                # Shape: (batch_size, n, d)
                if Q is not None:
                    z = torch.randn(current_batch_size, n, subspace_dim, device=device)
                    random_vectors = z @ Q.T  # (batch_size, n, d)
                else:
                    random_vectors = torch.randn(current_batch_size, n, d, device=device)
                proj = (random_vectors * K_hat.unsqueeze(0)).sum(dim=-1, keepdim=True)
                orthogonal = random_vectors - proj * K_hat.unsqueeze(0)
                orthogonal = F.normalize(orthogonal, p=2, dim=-1, eps=1e-12)

                # Construct perturbed input at angle theta from k_hat_i
                # cos_theta: (n,) -> (1, n, 1), sin_theta: (n,) -> (1, n, 1)
                perturbed = (
                    cos_theta.view(1, n, 1) * K_hat.unsqueeze(0)
                    + sin_theta.view(1, n, 1) * orthogonal
                )  # (batch_size, n, d), already unit norm

                # Run through model: (batch_size, n, d)
                perturbed_flat = perturbed.reshape(-1, d)
                outputs = model(perturbed_flat).view(current_batch_size, n, d)

                # Compute scores: (batch_size, n, n) where [b, i, j] = f(...)^T v_j
                scores = torch.bmm(
                    outputs, V_t.unsqueeze(0).expand(current_batch_size, -1, -1)
                )

                # Check if argmax_j f(...)^T v_j = i for each key i
                predicted = torch.argmax(scores, dim=-1)  # (batch_size, n)
                expected = (
                    torch.arange(n, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand_as(predicted)
                )
                correct_per_key = predicted == expected  # (batch_size, n)

                # A key passes only if ALL perturbations in this batch are correct
                batch_all_correct = correct_per_key.all(dim=0)  # (n,)
                key_all_correct &= batch_all_correct

            return key_all_correct

    # Parallel binary search across all keys
    low = torch.zeros(n, device=device)
    high = torch.full(
        (n,), min(float(max_epsilon_angle), float(np.pi)), device=device
    )
    best_valid = torch.zeros(n, device=device)

    while (high - low).max() > tolerance:
        mid = (low + high) / 2.0

        # Test all keys at their current mid angles
        passes = test_keys_at_angles(mid)

        # Update bounds based on results
        # If key passes: best_valid = mid, low = mid (try larger)
        # If key fails: high = mid (try smaller)
        best_valid = torch.where(passes, mid, best_valid)
        low = torch.where(passes, mid, low)
        high = torch.where(passes, high, mid)

    # Return the raw distribution of per-key maximum angles as a NumPy array
    return best_valid.detach().cpu().numpy()


def compute_mixer_input_margin_l2_distribution(
    model: torch.nn.Module,
    input_keys: torch.Tensor,
    sorted_value_matrix: torch.Tensor,
    num_samples: int = 5_000,
    max_epsilon_magnitude: float = 10.0,
    batch_size: int = 100,
    device: str | None = None,
    tolerance: float = 1e-4,
    noise_subspace: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Compute the per-key maximum epsilon magnitude such that decoding is correct, and return the
    full distribution of these per-key maximum margins.

    This mirrors `compute_mixer_input_margin_l2`, but instead of returning
    quantiles of the per-key maximum margins, it returns the raw distribution as
    a NumPy array of shape (n_keys,).

    Args:
        model: The MLP or linear model
        input_keys: (n, d) tensor of input key embeddings k_i
        sorted_value_matrix: (n, d) tensor of sorted value embeddings v_j
        num_samples: Number of perturbations to sample per magnitude per key
        max_epsilon_magnitude: Maximum magnitude to search
        batch_size: Batch size for processing perturbations
        device: Device to use
        tolerance: Tolerance for binary search
        noise_subspace: Optional (m, d) tensor of vectors spanning the subspace
            to which noise is restricted. If None, noise is sampled from all of R^d.

    Returns:
        NumPy array of per-key maximum margins (epsilon magnitudes)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    K = input_keys.to(device)
    V = sorted_value_matrix.to(device)
    n, d = K.shape

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    V_t = V.T  # (d, n)

    # Precompute orthonormal basis for noise subspace if provided
    if noise_subspace is not None:
        # noise_subspace: (m, d) -> Q: (d, r) where r = rank
        Q, _ = torch.linalg.qr(noise_subspace.T.to(device))
        subspace_dim = Q.shape[1]
    else:
        Q = None
        subspace_dim = d

    def test_keys_at_magnitudes(magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Test each key at its corresponding epsilon magnitude.

        Args:
            magnitudes: (n,) tensor of magnitudes to test, one per key

        Returns:
            (n,) boolean tensor - True if key i passes all perturbations at magnitudes[i]
        """
        with torch.no_grad():
            # Track correct predictions per key across all perturbations
            key_all_correct = torch.ones(n, device=device, dtype=torch.bool)

            # Test all perturbations in batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                current_batch_size = batch_end - batch_start

                # Sample random directions (unit vectors) - one per (perturbation, key)
                # Shape: (batch_size, n, d)
                if Q is not None:
                    z = torch.randn(current_batch_size, n, subspace_dim, device=device)
                    random_vectors = z @ Q.T  # (batch_size, n, d) - in subspace
                else:
                    random_vectors = torch.randn(current_batch_size, n, d, device=device)
                directions = F.normalize(random_vectors, p=2, dim=-1)

                # Scale each key's perturbations by its own magnitude
                # magnitudes: (n,) -> (1, n, 1)
                epsilons = directions * magnitudes.view(1, n, 1)  # (batch_size, n, d)

                # Add epsilon to each key: (batch_size, n, d)
                perturbed_inputs = K.unsqueeze(0) + epsilons

                # Run through model: (batch_size, n, d)
                perturbed_flat = perturbed_inputs.reshape(-1, d)
                outputs = model(perturbed_flat).view(current_batch_size, n, d)

                # Compute scores: (batch_size, n, n) where [b, i, j] = f(k_i + eps_b)^T v_j
                scores = torch.bmm(outputs, V_t.unsqueeze(0).expand(current_batch_size, -1, -1))

                # Check if argmax_j f(k_i + eps)^T v_j = i for each key i
                predicted = torch.argmax(scores, dim=-1)  # (batch_size, n)
                expected = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand_as(predicted)
                correct_per_key = predicted == expected  # (batch_size, n)

                # A key passes only if ALL perturbations in this batch are correct
                batch_all_correct = correct_per_key.all(dim=0)  # (n,)
                key_all_correct &= batch_all_correct

            return key_all_correct

    # Parallel binary search across all keys
    low = torch.zeros(n, device=device)
    high = torch.full((n,), float(max_epsilon_magnitude), device=device)
    best_valid = torch.zeros(n, device=device)

    while (high - low).max() > tolerance:
        mid = (low + high) / 2.0

        # Test all keys at their current mid magnitudes
        passes = test_keys_at_magnitudes(mid)

        # Update bounds based on results
        # If key passes: best_valid = mid, low = mid (try larger)
        # If key fails: high = mid (try smaller)
        best_valid = torch.where(passes, mid, best_valid)
        low = torch.where(passes, mid, low)
        high = torch.where(passes, high, mid)

    # Return the raw distribution of per-key maximum margins as a NumPy array
    return best_valid.detach().cpu().numpy()
