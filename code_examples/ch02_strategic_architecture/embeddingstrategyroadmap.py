# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingStrategyRoadmap:
    """Phased roadmap with measurable milestones"""

    def __init__(self, vision, current_maturity_level):
        self.vision = vision
        self.current_level = current_maturity_level
        self.milestones = []

    def define_phases(self):
        """Create phased roadmap from current state to vision"""

        return {
            'phase_1_foundation': {
                'duration_months': 6,
                'objectives': [
                    'Establish embedding infrastructure (vector DB, training pipeline)',
                    'Deploy first production use case',
                    'Build initial embedding team',
                    'Create data pipelines for embedding generation'
                ],
                'success_criteria': {
                    'technical': [
                        'Vector DB serving embeddings with acceptable latency',
                        'Training pipeline producing embeddings for new items',
                        'Monitoring and observability in place'
                    ],
                    'business': [
                        'First use case showing measurable improvement over baseline',
                        'Executive stakeholder buy-in secured',
                        'Budget approved for Phase 2'
                    ]
                },
                'team_size': 'Small team of ML engineers and infrastructure specialists'
            },

            'phase_2_expansion': {
                'duration_months': 12,
                'objectives': [
                    'Scale to larger embedding collections across multiple use cases',
                    'Develop first custom embedding model',
                    'Establish MLOps practices (versioning, AB testing, monitoring)',
                    'Build multi-modal capabilities (text + images)'
                ],
                'success_criteria': {
                    'technical': [
                        'Serving embeddings across multiple production use cases',
                        'Custom model outperforms off-the-shelf baseline',
                        'AB testing infrastructure validates improvements',
                        'Zero-downtime deployment process'
                    ],
                    'business': [
                        'Multiple use cases in production with documented ROI',
                        'Measurable aggregate business impact',
                        'Embedding platform adopted by multiple internal teams'
                    ]
                },
                'team_size': 'Expanded team with specialized roles'
            },

            'phase_3_transformation': {
                'duration_months': 18,
                'objectives': [
                    'Scale to very large embedding collections',
                    'Embedding platform becomes core infrastructure',
                    'Advanced multi-modal (text, images, audio, structured data)',
                    'Real-time embedding updates and retraining'
                ],
                'success_criteria': {
                    'technical': [
                        'Large-scale embeddings served globally',
                        'Multi-region deployment with low latency',
                        'Real-time incremental updates',
                        'Advanced capabilities (semantic search, RAG, anomaly detection)'
                    ],
                    'business': [
                        'Widespread production deployment across organization',
                        'Significant aggregate business impact',
                        'Documented competitive advantage in core product',
                        'Customer-facing features powered by embeddings'
                    ]
                },
                'team_size': 'Large cross-functional organization'
            },

            'phase_4_leadership': {
                'duration_months': 24,
                'objectives': [
                    'Trillion-scale embedding infrastructure',
                    'Proprietary embedding methods',
                    'Organization-wide embedding adoption',
                    'Ecosystem and platform effects'
                ],
                'success_criteria': {
                    'technical': [
                        'Trillion-scale embeddings served globally',
                        'Proprietary methods published/patented',
                        'Industry-leading performance benchmarks',
                        'Open-source contributions establish thought leadership'
                    ],
                    'business': [
                        'Widespread production use cases throughout organization',
                        'Substantial aggregate business impact',
                        'Embeddings are core competitive moat',
                        'New business models enabled by embedding capabilities'
                    ]
                },
                'team_size': 'Dedicated embedding platform organization'
            }
        }

# Example: E-commerce company currently at Level 1
roadmap = EmbeddingStrategyRoadmap(
    vision="Every product discovery interaction powered by embeddings",
    current_maturity_level=1
)

phases = roadmap.define_phases()
