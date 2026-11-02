import numpy as np


def conflate_probabilities(source_probs: list, class_weights: np.ndarray = None, eps: float = 1e-9) -> np.ndarray:
    '''
    fuse multiple class-probability vectors (e.g. a video classifier's play-type
    output and a Markov Chain's next-play-type prediction) via entry-wise
    multiplication, per Section 4.3 / Fig. 11 of the Smart Stadium paper

    Combined in log-space so per-class weights act as a multiplicative
    correction: fused ∝ exp(sum(log(p_i)) + log(class_weights))

    Args:
        source_probs: list of probability vectors, all the same length, one per signal
        class_weights: optional per-class weights (learned via fit_conflation_weights)
                       applied on top of the elementwise product, to correct for any
                       consistent bias between the sources
        eps: numerical floor so log() never sees zero

    Returns:
        the fused, renormalized probability vector
    '''
    log_fused = np.sum([np.log(p + eps) for p in source_probs], axis=0)
    if class_weights is not None:
        log_fused = log_fused + np.log(class_weights + eps)

    log_fused = log_fused - log_fused.max()  # numerical stability before exp
    fused = np.exp(log_fused)
    return fused / fused.sum()


def fit_conflation_weights(samples: list, num_classes: int, epochs: int = 500, lr: float = 0.5,
                            eps: float = 1e-9) -> np.ndarray:
    '''
    learn per-class weights for conflate_probabilities via gradient descent on
    cross-entropy loss -- the "final Neural Network that optimally calculates
    the specific weights" described in Section 4.3. Equivalent to fitting a
    per-class bias on top of the two sources' summed log-probabilities
    (a single-layer softmax model), then converting back to a multiplicative weight.

    Args:
        samples: list of (cnn_probs, mc_probs, true_label_idx) tuples
        num_classes: number of play-type classes
        epochs: gradient descent iterations
        lr: learning rate
        eps: numerical floor so log() never sees zero

    Returns:
        learned per-class weight vector, to pass as class_weights to conflate_probabilities
    '''
    beta = np.zeros(num_classes)  # log-weights; start neutral (weight = 1 for every class)
    labels = np.eye(num_classes)

    for _ in range(epochs):
        grad = np.zeros(num_classes)
        for cnn_probs, mc_probs, label in samples:
            logits = np.log(cnn_probs + eps) + np.log(mc_probs + eps) + beta
            logits = logits - logits.max()
            fused = np.exp(logits)
            fused /= fused.sum()
            grad += fused - labels[label]
        beta -= lr * grad / len(samples)

    return np.exp(beta)


if __name__ == '__main__':
    pass
