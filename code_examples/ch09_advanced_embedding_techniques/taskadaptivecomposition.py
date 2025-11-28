import torch
import torch.nn as nn

# Code from Chapter 08
# Book: Embeddings at Scale


class TaskAdaptiveComposition(nn.Module):
    """
    Learn task-specific composition strategies

    Example: Product embeddings
    - For search: weight title, category highly
    - For recommendations: weight reviews, purchase history highly
    - For fraud detection: weight price anomalies, seller reputation highly

    Single compositional model, multiple task-specific heads
    """

    def __init__(self, component_dims, tasks, output_dim=256):
        """
        Args:
            component_dims: Dict of component dimensions
            tasks: List of task names ['search', 'recommendation', 'fraud']
            output_dim: Output dimension
        """
        super().__init__()
        self.tasks = tasks

        # Project components to common space
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, output_dim) for name, dim in component_dims.items()}
        )

        # Task-specific attention for composition
        self.task_attention = nn.ModuleDict(
            {
                task: nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)
                for task in tasks
            }
        )

        # Task-specific queries (what each task "looks for")
        self.task_queries = nn.ParameterDict(
            {task: nn.Parameter(torch.randn(1, 1, output_dim)) for task in tasks}
        )

    def forward(self, component_embeddings, task="search"):
        """
        Compose embeddings for specific task

        Args:
            component_embeddings: Dict of component embeddings
            task: Which task this is for

        Returns:
            Task-specific compositional embedding
        """
        # Project components
        projected = {
            name: self.projections[name](emb) for name, emb in component_embeddings.items()
        }

        # Stack for attention
        stacked = torch.stack(list(projected.values()), dim=1)
        batch_size = stacked.shape[0]

        # Get task-specific query
        query = self.task_queries[task].expand(batch_size, -1, -1)

        # Task-specific attention
        composed, _ = self.task_attention[task](query=query, key=stacked, value=stacked)

        return composed.squeeze(1)
