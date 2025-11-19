# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingAccessControl:
    """Access control for embedding systems"""

    def __init__(self):
        self.access_policies = {}
        self.audit_log = []

    def define_access_policy(self, embedding_collection, policy):
        """Define who can access which embeddings"""
        self.access_policies[embedding_collection] = {
            'read_access': policy.get('read_access', []),  # User/role list
            'write_access': policy.get('write_access', []),
            'delete_access': policy.get('delete_access', []),
            'data_sensitivity': policy.get('sensitivity', 'public'),  # public, internal, confidential, restricted
            'retention_policy': policy.get('retention_days', 365),
            'encryption_required': policy.get('encryption', True),
            'audit_required': policy.get('audit', True)
        }

    def check_access(self, user, embedding_collection, operation):
        """Check if user has access"""
        policy = self.access_policies.get(embedding_collection)

        if policy is None:
            return False  # Deny by default

        # Check appropriate access list
        access_list = policy[f'{operation}_access']

        has_access = (
            user['id'] in access_list or
            any(role in access_list for role in user['roles'])
        )

        # Log access attempt
        if policy['audit_required']:
            self.audit_log.append({
                'timestamp': datetime.now(),
                'user': user['id'],
                'collection': embedding_collection,
                'operation': operation,
                'granted': has_access
            })

        return has_access

    def encrypt_embeddings(self, embeddings, encryption_key):
        """Encrypt embeddings at rest"""
        # Use homomorphic encryption for privacy-preserving search
        # Or standard encryption if only storing
        from cryptography.fernet import Fernet

        fernet = Fernet(encryption_key)
        encrypted = fernet.encrypt(embeddings.tobytes())

        return encrypted
