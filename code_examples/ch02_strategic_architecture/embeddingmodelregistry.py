# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingModelRegistry:
    """Central registry for embedding models"""

    def register_model(self, model_metadata):
        """Register new embedding model with governance metadata"""
        required_fields = [
            'model_id',
            'model_architecture',
            'training_data_sources',
            'training_date',
            'owner',
            'use_cases',
            'approval_status',
            'bias_audit_results',
            'performance_metrics',
            'deployment_restrictions'
        ]

        # Validate all required metadata present
        for field in required_fields:
            if field not in model_metadata:
                raise ValueError(f"Missing required field: {field}")

        # Store in registry
        self.registry[model_metadata['model_id']] = {
            **model_metadata,
            'registration_timestamp': datetime.now(),
            'version': self.get_next_version(model_metadata['model_id']),
            'audit_trail': []
        }

        # Trigger approval workflow
        self.initiate_approval_workflow(model_metadata['model_id'])

    def approve_model_for_use_case(self, model_id, use_case, approver):
        """Approve model for specific use case"""
        model = self.registry[model_id]

        # Log approval
        model['audit_trail'].append({
            'timestamp': datetime.now(),
            'action': 'approved',
            'use_case': use_case,
            'approver': approver,
            'approval_reason': f"Model approved for {use_case}"
        })

        # Update approval status
        if 'approved_use_cases' not in model:
            model['approved_use_cases'] = []
        model['approved_use_cases'].append(use_case)

    def audit_model_usage(self, model_id):
        """Audit trail for model usage"""
        model = self.registry[model_id]

        return {
            'model_id': model_id,
            'version': model['version'],
            'approved_use_cases': model.get('approved_use_cases', []),
            'actual_deployments': self.get_actual_deployments(model_id),
            'audit_trail': model['audit_trail'],
            'last_bias_audit': model['bias_audit_results']['timestamp'],
            'last_performance_review': model['performance_metrics']['timestamp']
        }
