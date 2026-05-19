import torch


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        # Check for out-of-bounds indices and clamp them
        max_nodes = p.size(0) - 1  # Maximum valid index
        valid_indices = torch.clamp(batch_n_nodes, 0, max_nodes)
        
        # Warn if there are any out-of-bounds values (but only once to avoid spam)
        if torch.any(batch_n_nodes > max_nodes):
            if not hasattr(self, '_warned_max'):
                print(f"Warning: Found node counts exceeding training distribution max ({max_nodes}). "
                      f"Max count in batch: {batch_n_nodes.max().item()}. Using probability of max valid count.")
                self._warned_max = True
        if torch.any(batch_n_nodes < 0):
            if not hasattr(self, '_warned_min'):
                print(f"Warning: Found negative node counts. Min count in batch: {batch_n_nodes.min().item()}. "
                      f"Using probability of 0 nodes.")
                self._warned_min = True

        probas = p[valid_indices]
        log_p = torch.log(probas + 1e-6)
        return log_p
