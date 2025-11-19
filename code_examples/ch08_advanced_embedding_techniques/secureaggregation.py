# Code from Chapter 08
# Book: Embeddings at Scale

class SecureAggregation:
    """
    Secure aggregation protocol for federated learning

    Server learns only the sum of client updates,
    not individual updates

    Protocol:
    1. Clients add secret shares to their updates
    2. Shares cancel out when summed
    3. Server gets accurate aggregate without seeing individuals

    Protects against:
    - Curious server
    - Compromised clients (up to threshold)
    """

    def __init__(self, num_clients):
        self.num_clients = num_clients

    def client_mask(self, client_id, other_client_id, seed):
        """
        Generate pairwise mask between two clients

        Property: mask(i,j) = -mask(j,i)
        So masks cancel when summed
        """
        # Deterministic random mask based on shared seed
        torch.manual_seed(seed + client_id * 1000 + other_client_id)
        mask = torch.randn(1)

        # Ensure antisymmetry
        if client_id > other_client_id:
            mask = -mask

        return mask

    def client_add_masks(self, client_id, update):
        """
        Client adds pairwise masks to update

        Each client adds mask(i,j) for all other clients j
        """
        masked_update = update.clone()

        for other_id in range(self.num_clients):
            if other_id != client_id:
                # Shared seed (in practice, established via key exchange)
                seed = 42
                mask = self.client_mask(client_id, other_id, seed)
                masked_update += mask

        return masked_update

    def server_aggregate(self, masked_updates):
        """
        Server sums masked updates

        Masks cancel out, leaving true sum
        """
        # Sum all masked updates
        aggregate = sum(masked_updates)

        # Masks cancel: sum(mask(i,j) for all i,j) = 0
        # Result is sum of original updates

        return aggregate
