# Code from Chapter 08
# Book: Embeddings at Scale

def embed_medical_ontology():
    """
    Medical ontology example: Disease hierarchies

    ICD-10 codes have 14,000+ diseases organized hierarchically
    Hyperbolic embeddings in 10-20 dimensions outperform
    Euclidean embeddings in 300-500 dimensions
    """
    # Example: Simplified disease taxonomy
    disease_taxonomy = {
        # Cardiovascular diseases
        'myocardial_infarction': 'ischemic_heart_disease',
        'angina': 'ischemic_heart_disease',
        'ischemic_heart_disease': 'cardiovascular_disease',

        'atrial_fibrillation': 'arrhythmia',
        'ventricular_tachycardia': 'arrhythmia',
        'arrhythmia': 'cardiovascular_disease',

        # Respiratory diseases
        'pneumonia': 'lower_respiratory_infection',
        'bronchitis': 'lower_respiratory_infection',
        'lower_respiratory_infection': 'respiratory_disease',

        'asthma': 'chronic_respiratory_disease',
        'copd': 'chronic_respiratory_disease',
        'chronic_respiratory_disease': 'respiratory_disease',
    }

    trainer = HierarchicalEmbeddingTrainer(
        disease_taxonomy,
        embedding_dim=10,
        curvature=1.0
    )

    trainer.train(num_epochs=2000, verbose=True)

    return trainer
