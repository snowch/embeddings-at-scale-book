# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingComplianceFramework:
    """Ensure regulatory compliance"""

    def gdpr_compliance_check(self, embedding_system):
        """Verify GDPR compliance"""
        compliance = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }

        # Right to erasure
        if not embedding_system.supports_deletion():
            compliance['compliant'] = False
            compliance['violations'].append("Cannot delete individual embeddings (Right to Erasure)")

        # Data minimization
        if embedding_system.stores_raw_data_with_embeddings():
            compliance['recommendations'].append(
                "Consider storing only embeddings, not raw data (Data Minimization)"
            )

        # Purpose limitation
        if not embedding_system.has_documented_purposes():
            compliance['compliant'] = False
            compliance['violations'].append("No documented data processing purposes")

        # Transparency
        if not embedding_system.can_explain_decisions():
            compliance['recommendations'].append(
                "Add explainability for automated decisions (Transparency)"
            )

        # Data protection by design
        if not embedding_system.has_privacy_controls():
            compliance['compliant'] = False
            compliance['violations'].append(
                "No privacy controls (Data Protection by Design)"
            )

        return compliance

    def hipaa_compliance_check(self, embedding_system):
        """Verify HIPAA compliance for healthcare"""
        compliance = {
            'compliant': True,
            'violations': []
        }

        # PHI Protection
        if not embedding_system.encrypts_data_at_rest():
            compliance['compliant'] = False
            compliance['violations'].append("PHI not encrypted at rest")

        if not embedding_system.encrypts_data_in_transit():
            compliance['compliant'] = False
            compliance['violations'].append("PHI not encrypted in transit")

        # Access controls
        if not embedding_system.has_role_based_access():
            compliance['compliant'] = False
            compliance['violations'].append("No role-based access controls")

        # Audit trails
        if not embedding_system.maintains_audit_trails():
            compliance['compliant'] = False
            compliance['violations'].append("No audit trails for PHI access")

        # De-identification
        if embedding_system.uses_identifiable_phi():
            compliance['recommendations'].append(
                "Consider using de-identified data where possible"
            )

        return compliance
