# AI Coding Agent Instructions for Drug-Target Interaction Prediction Project

## Project Overview
This is an MSc research project developing Graph Neural Networks (GNNs) for predicting kinase inhibitor off-target effects. The project combines molecular graph representations, protein embeddings, and explainable AI to achieve state-of-the-art drug-target interaction prediction.

## Architecture & Data Flow

### Sequential Processing Pipeline
The project follows a numbered step-by-step workflow (`steps/step*.ipynb`):
1. **Data Collection**: Fetch kinase inhibitor data from ChEMBL database
2. **SMILES Processing**: Convert molecule IDs to SMILES representations
3. **Protein Sequences**: Fetch FASTA sequences for kinase targets
4. **Protein Embeddings**: Generate efficient one-hot encodings (115 features) OR ESM transformer embeddings
5. **Molecular Graphs**: Convert SMILES to PyTorch Geometric graphs with RDKit
6. **Dataset Pairing**: Create (drug_graph, target_embedding) → binding_label training pairs
7. **Model Training**: Train GNN models with accuracy maximization strategy

### Key Data Structures
- **Molecular Graphs**: PyTorch Geometric `Data` objects stored in `data/graphs/*.pt`
  - Node features: atomic number, formal charge, aromaticity, hybridization, degree, hydrogens
  - Edge features: bond type, conjugation, ring membership
  - Graph-level: molecular weight, logP, Morgan fingerprint (2048-bit)
- **Protein Embeddings**: Two approaches available
  - Efficient one-hot: 115 interpretable features (composition + positional)
  - ESM: 1280-dimensional transformer embeddings from Facebook's protein model
- **Training Pairs**: CSV format with `drug_id`, `target_id`, `label` (binary classification)

## Critical Patterns & Conventions

### Protein Embedding Strategy
The project uses **Efficient One-Hot Encoding** as the primary approach:
```python
# Position-aware features: N-terminal, C-terminal, central regions
# Physicochemical groupings: hydrophobic, polar, charged, special
# Final output: 115 interpretable features vs 21,000+ traditional one-hot
```
Use this over ESM embeddings unless specifically working with generalizability studies.

### Graph Generation Pattern
```python
# Standard pattern for molecular graphs
def smiles_to_graph(smiles, mol_id):
    # Always include: atom features, bond features, molecular properties, fingerprints
    return Data(x=node_features, edge_index=edges, edge_attr=edge_features, 
                mol_weight=mw, logp=logp, fingerprint=fp, mol_id=mol_id)
```

### File Naming Conventions
- Step outputs: `data/step{N}_{description}.csv`
- Molecular graphs: `data/graphs/{CHEMBL_ID}.pt`
- Models: Should be saved to `results/` or `models/` directories
- Notebooks: `steps/step{N}_{description}.ipynb` for main pipeline

### ChEMBL Data Handling
- Target IDs: Format like `CHEMBL1862` 
- Molecule IDs: Format like `CHEMBL100076`
- Activity cutoff: 1000 nM (1 µM) for binary classification
- Protein ID format in embeddings: `P00519|CHEMBL1862|Tyrosine-protein_kinase_ABL1`

## Model Architecture Priorities

### Training Strategy
Use **accuracy maximization** instead of traditional loss minimization:
```python
# Focus on classification accuracy as primary metric
# Direct optimization of accuracy achieves 81.01% vs ~70% with loss minimization
# Validated as superior approach for biomedical classification tasks
```

### Model Variants
Five model architectures are evaluated:
1. MLP Baseline
2. Original GraphSAGE  
3. Improved GraphSAGE
4. Performance Booster
5. **Accuracy Optimized** (best performing: AUC 0.8859)

## Development Workflows

### Running the Full Pipeline
Execute notebooks sequentially: `step1` → `step2` → ... → `step7`
Each step depends on outputs from previous steps.

### Adding New Features
- Molecular features: Modify `smiles_to_graph()` in step5
- Protein features: Extend embedding generation in step4/4.1
- Always update feature dimensions in subsequent training steps

### Generalizability Testing
Use `generalizability/` notebooks for external validation:
- DAVIS and KIBA datasets for benchmarking
- Compare one-hot vs ESM embeddings
- Feature sensitivity analysis in step4

## Dependencies & Environment

### Core Libraries
- `torch` + `torch_geometric`: Graph neural networks
- `rdkit`: Molecular graph generation and descriptors
- `transformers` + `esm`: Protein language models (ESM-2)
- `chembl_webresource_client`: ChEMBL API access
- `biopython`: FASTA sequence processing

### Data Sources
- ChEMBL database: Primary source for kinase inhibitor data
- External benchmarks: DAVIS/KIBA datasets in `generalizability/data/external/`

## Debugging & Validation

### Common Issues
- **Missing graphs**: Check `data/graphs/` directory exists and contains `.pt` files
- **Protein embedding mismatches**: Verify ChEMBL ID format consistency
- **Memory issues**: Molecular graphs can be large; batch processing recommended
- **API limits**: ChEMBL requests may need rate limiting

### Validation Patterns
- Always check data availability before pairing (step6 pattern)
- Validate graph generation with `mol is not None` checks
- Use `drop_duplicates()` when creating training pairs
- Filter activities by type: `["IC50", "Ki", "Kd"]` only

## Explainability Integration

### GNNExplainer Usage
The project integrates explainable AI for biological validation:
- Identifies key molecular features (ATP-binding domains, hinge regions)
- Validates against known kinase pharmacology
- Results stored in `explanations/` directory

When implementing explainability:
- Focus on interpretable molecular substructures
- Cross-reference with established structure-activity relationships
- Document biological relevance of identified features
