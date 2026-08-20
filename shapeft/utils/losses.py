import torch

class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w if w!=0 else 0 for w in distribution]
        self.ignore_index = ignore_index
        self.register_buffer("loss_weights", torch.tensor(self.weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return torch.nn.functional.cross_entropy(
            logits, target, ignore_index=self.ignore_index, weight=self.loss_weights
        )
