# Tox21 Multi-Target Toxicity Prediction Pipeline


---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Why ROC-AUC?](#why-roc-auc)
4. [Feature Engineering — 11 Layers](#feature-engineering--11-layers)
5. [Preprocessing & Imputation](#preprocessing--imputation)
6. [Model Architecture](#model-architecture)
   - [Stage 1 — XGBoost Baseline](#stage-1--xgboost-baseline-with-5-fold-cross-validation)
   - [Stage 2 — Cross-Endpoint Meta-Features](#stage-2--cross-endpoint-meta-feature-construction)
   - [Stage 3 — Endpoint-Aware Adaptive Ensemble](#stage-3--endpoint-aware-adaptive-ensemble)
   - [Stage 4 — Per-Endpoint Threshold Optimisation](#stage-4--per-endpoint-decision-threshold-optimisation)
   - [Stage 5 — Feature Importance Analysis](#stage-5--feature-importance-analysis)
7. [Prediction Interface](#prediction-interface)
8. [Results & Outputs](#results--outputs)
9. [Repository Structure](#repository-structure)
10. [Setup & Reproduction](#setup--reproduction)
11. [Design Decisions & Justifications](#design-decisions--justifications)

---

## Problem Statement

The [Tox21 dataset](https://tripod.nih.gov/tox21/challenge/) is a benchmark challenge from the NIH/EPA/FDA designed to predict the toxicity of chemical compounds against 12 biological targets, spanning nuclear receptors (NR) and stress-response (SR) pathways. Given only the SMILES string of a molecule, the goal is to predict binary toxicity labels (1 = toxic, 0 = non-toxic) for each of the 12 endpoints simultaneously.

**The 12 Tox21 Endpoints:**

| Prefix | Endpoint | Biological Target |
|--------|----------|-------------------|
| NR | NR-AR | Androgen Receptor (full length) |
| NR | NR-AR-LBD | Androgen Receptor Ligand Binding Domain |
| NR | NR-AhR | Aryl hydrocarbon Receptor |
| NR | NR-Aromatase | Aromatase Enzyme |
| NR | NR-ER | Estrogen Receptor Alpha (full length) |
| NR | NR-ER-LBD | Estrogen Receptor Alpha LBD |
| NR | NR-PPAR-gamma | Peroxisome Proliferator Activated Receptor Gamma |
| SR | SR-ARE | Antioxidant Response Element |
| SR | SR-ATAD5 | Genotoxicity / DNA Damage Marker |
| SR | SR-HSE | Heat Shock Element |
| SR | SR-MMP | Mitochondrial Membrane Potential |
| SR | SR-p53 | p53 Tumour Suppressor Pathway |

The dataset is characterised by two major challenges:

1. **Missing labels (masking):** Not every compound was tested against every endpoint. Up to 30–40% of labels per endpoint are `NaN`. Models must be trained only on labelled compounds per endpoint.
2. **Severe class imbalance:** Toxic compounds are the minority class across all endpoints, with positive rates ranging roughly from 3% to 25%. The ratio of safe-to-toxic compounds (`scale_pos_weight`) can exceed 20:1 for some endpoints.

---

## Dataset

The full Tox21 dataset (`tox21.csv`) contains ~7,800 unique chemical compounds represented as SMILES strings, with 12 binary toxicity label columns. The uploaded `codecuresl.csv` contains the engineered physicochemical and structural descriptor columns used in the pipeline (Morgan fingerprint columns of 2048 dimensions at radius=2 with chirality were excluded from the upload due to file size constraints but are generated in-pipeline from SMILES).

**Column groups in `codecuresl.csv`:**

- `smiles` — canonical SMILES string
- `MW, logP, QED, TPSA, HBD, HBA, RotBonds, ArRings, FracCSP3, HeavyAtoms` — core RDKit physicochemical descriptors
- `lip_pass, lip_violations, logP_x_MW, TPSA_per_MW, HBD_HBA_sum, logMW` — engineered Lipinski-derived features
- `zinc_SAS, drug_like, fragment_like, lead_like` — ZINC drug-likeness flags
- `chiral_n_R, chiral_n_S, chiral_total, has_chirality, chiral_RS_ratio` — stereochemistry features
- `total_charge, n_pos_atoms, n_neg_atoms, max_atom_charge, charge_range` — formal charge features
- `NR-AR` through `SR-p53` — the 12 binary toxicity labels (with NaN for untested compounds)

**Invalid SMILES filtering:** Before any feature computation, all SMILES strings are parsed by RDKit (`Chem.MolFromSmiles`). Rows yielding `None` (invalid SMILES) are dropped.

---

## Why ROC-AUC?

ROC-AUC (Area Under the Receiver Operating Characteristic Curve) is the correct primary metric for this problem for several compounding reasons:

**1. Severe and variable class imbalance.** Accuracy is meaningless when a model that always predicts "non-toxic" achieves 93%+ accuracy on an endpoint with 7% positive rate. ROC-AUC measures the model's ability to rank toxic compounds above non-toxic ones, independent of class distribution. A random classifier always scores 0.5 regardless of imbalance; a perfect classifier scores 1.0.

**2. Missing labels require per-endpoint evaluation.** Each endpoint has a different subset of labelled compounds and a different positive rate. ROC-AUC is computed per endpoint on its own labelled subset, making it a consistent, comparable metric across all 12 tasks despite their different sizes and imbalances.

**3. Threshold independence.** The optimal decision threshold for flagging a compound as toxic varies per endpoint and per downstream use case (a drug safety screen may prioritise recall over precision). ROC-AUC evaluates the entire score distribution rather than fixing a threshold, which is why threshold optimisation is handled separately in Stage 4 using Youden's J statistic.

**4. Standard for Tox21 and cheminformatics benchmarks.** The original Tox21 challenge and all major subsequent publications on this dataset (DeepChem, ChemProp, AttentiveFP, etc.) use mean ROC-AUC across the 12 endpoints as the primary benchmark metric. Using it enables direct comparison with the literature.

**5. Interpretability of ensemble improvement.** The delta in ROC-AUC between models directly quantifies ranking improvement, making it intuitive to communicate per-endpoint model gains.

**Caveat:** For deployed drug safety applications, recall (sensitivity) at a specified threshold is ultimately more actionable than AUC, which is why Stage 4 explicitly optimises recall via Youden's J threshold tuning on top of the AUC-optimised models.

---

## Feature Engineering — 11 Layers

A total of **11 distinct feature layers** were constructed from SMILES strings using RDKit, covering structural fingerprints, physicochemical descriptors, 3D conformer descriptors, and engineered domain features.

### Layer 1A — Morgan ECFP4 Fingerprints (2048 bits)

Extended Connectivity Fingerprints with radius=2 (ECFP4), generated with chirality encoding enabled. Each bit encodes the presence of a circular atom environment within a radius of 2 bonds from a given atom. At 2048 bits, this gives sufficient resolution to distinguish diverse chemical scaffolds.

- **Chirality:** Enabled via `includeChirality=True` — critical for NR-AR and NR-ER, where stereoisomers can have opposite hormonal activity.
- **Post-processing:** VarianceThreshold at `p*(1-p)` for `p=0.001` removes bits that are ON in fewer than 0.1% or more than 99.9% of molecules (near-constant, uninformative bits).

### Layer 1B — Atom Pair Fingerprints (1024 bits)

Hashed atom pair fingerprints encode pairs of heavy atoms and the shortest topological distance between them. They capture long-range atom interactions that circular fingerprints at radius=2 miss. Chirality-aware version used.

### Layer 1C — Topological Torsion Fingerprints (1024 bits)

Encode sequences of 4 consecutively bonded atoms and their topological properties. Particularly informative for flexible chain-like molecules and conformational flexibility encoding. Chirality-aware version used.

### Layer 2 — MACCS Structural Keys (167 bits)

A pre-defined dictionary of 167 SMARTS-based structural patterns (ring systems, functional groups, specific bond types). Unlike Morgan fingerprints, MACCS keys are interpretable — each bit has a known chemical meaning. They complement the data-driven Morgan encoding with expert-defined chemical knowledge.

### Layer 3 — Core RDKit Physicochemical Descriptors (10 features)

The canonical Lipinski-era ADME descriptors: molecular weight (MW), lipophilicity (logP), drug-likeness score (QED), topological polar surface area (TPSA), hydrogen bond donors (HBD), hydrogen bond acceptors (HBA), rotatable bonds, aromatic rings, fraction of sp3 carbons (FracCSP3), and heavy atom count. Computed directly from the 2D graph (no conformer needed).

### Layer 4 — Extended RDKit 2D Descriptor Suite (~200 features)

RDKit's full suite of ~200 additional 2D descriptors: VSA descriptors (MOE-style surface area partitioned by logP, charge, and H-bonding), Estate indices (electrochemical atom environments), connectivity indices (chi0–chi4, kappa shape indices), ring system complexity, and fragment counts for functional groups. Descriptors with more than 50% missing values or infinite values are dropped before imputation.

### Layer 5 — NR-ER SMARTS Fragment Counts (8 features)

Eight biologically motivated SMARTS patterns hand-crafted to encode the pharmacophore features of estrogen receptor binders: phenol groups, aliphatic hydroxyls, benzene rings, fused aromatic systems, steroid-like ABC ring systems, halogenated phenyl groups, primary/secondary amines, and non-ring carbonyls. These are substructure match counts (not binary), specifically targeted at improving NR-ER and NR-ER-LBD performance.

### Layer 6 — Engineered Lipinski & Interaction Features (6 features)

Domain-informed feature crosses: Lipinski rule-of-five compliance flag (`lip_pass`), integer count of Lipinski violations (`lip_violations`), `logP × MW` interaction term (combined lipophilicity-size space), `TPSA / MW` polarity density, total hydrogen bonding capacity (`HBD + HBA`), and log-transformed MW to reduce right skew.

### Layer 7 — ZINC Drug-Likeness Flags (4 features)

ZINC-style compound classification: `drug_like` (MW 150–500, logP −0.4 to 5.6), `fragment_like` (MW < 300, logP < 3), `lead_like` (MW 250–350, logP −1 to 3.5), and Synthetic Accessibility Score (`zinc_SAS`, estimated via RDKit SAS scorer or a structural heuristic fallback).

### Layer 8 — Chirality & Formal Charge Features (10 features)

Stereochemistry and ionisation state: R/S chiral centre counts, total stereocentre count, binary chirality flag, R/S ratio, total formal charge, count of positively and negatively charged atoms, maximum atom charge, and charge range. Critical for differentiating enantiomers in receptor-binding endpoints (NR-AR, NR-ER) and capturing ionisation state effects.

### Layer 9 — AUTOCORR3D Descriptors (80 features)

3D autocorrelation descriptors computed from MMFF94-optimised conformers (generated via ETKDGv3 embedding). AUTOCORR3D computes correlation functions of atomic properties (mass, polarisability, electronegativity, charge, lipophilicity) at discrete distance lags, encoding 3D shape and atom-property distribution that 2D fingerprints fundamentally cannot capture. Conformer embedding failures are handled gracefully via NaN insertion and subsequent KNN imputation.

### Layer 10 — WHIM 3D Descriptors (114 → filtered)

Weighted Holistic Invariant Molecular descriptors encode molecular 3D shape relative to principal axes, weighted by atomic properties. WHIM captures size, shape, symmetry, and density in a rotation-invariant form. Near-zero-variance WHIM features are removed by VarianceThreshold (threshold=0.01) post-computation.

### Layer 11 — Cross-Endpoint OOF Meta-Features (Stage 2, 12 features)

Out-of-fold XGBoost predictions from Stage 1 are used as additional features for Stage 3. These 12 meta-features encode the model's probabilistic toxicity estimate at each endpoint, allowing Stage 3 ensemble models to exploit known biological co-regulation patterns (e.g., NR-AR and NR-AR-LBD, or ER-related endpoints sharing scaffold features).

---

## Preprocessing & Imputation

### Missing Label Masking

Toxicity labels are missing (NaN) for many compounds per endpoint. The pipeline trains a **separate model per endpoint** using only the labelled subset for that endpoint, implemented via a boolean mask that selects rows where the label is not NaN. This is the correct way to handle multi-task label sparsity without imputing labels or leaking information.

### KNN Imputation for Continuous Features

The extended RDKit descriptors, 3D descriptors, and some physicochemical features can contain NaN values due to computation failures on edge-case molecules. These are imputed using **K-Nearest Neighbours Imputation** with `k=5` and distance-weighted averaging. KNN imputation is preferred over mean imputation because physicochemical properties are correlated (MW and HeavyAtoms, logP and FracCSP3, etc.), and KNN preserves inter-feature correlations when filling gaps — mean imputation would ignore this structure and introduce bias.

### VarianceThreshold for Fingerprint Bits

After concatenating all fingerprint layers (Morgan + Atom Pair + Topological Torsion + MACCS), a VarianceThreshold filter removes near-constant bits. The threshold `p*(1-p)` with `p=0.001` removes any bit ON in fewer than 0.1% or more than 99.9% of molecules.

### Class Imbalance — `scale_pos_weight`

For each endpoint, the ratio of negative to positive samples (`n_safe / n_toxic`) is computed and passed as `scale_pos_weight` to XGBoost and LightGBM. This reweights the loss function to penalise minority-class misclassifications proportionally more, without synthetic oversampling.

---

## Model Architecture

### Stage 1 — XGBoost Baseline with 5-Fold Cross-Validation

For each endpoint, stratified 5-fold cross-validation is run on the labelled subset using XGBoost with `scale_pos_weight` set per endpoint, `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, and `colsample_bytree=0.8`. Out-of-fold (OOF) predictions, per-fold AUC scores, and fold-level models are cached. A final XGBoost model is also trained on the full labelled data per endpoint for deployment.

XGBoost is used as the baseline because it handles sparse binary features (fingerprints) efficiently, natively supports sample weighting for imbalance, and has strong documented performance on molecular property prediction benchmarks.

### Stage 2 — Cross-Endpoint Meta-Feature Construction

The OOF probability scores from Stage 1 are assembled into a `(n_compounds × 12)` meta-feature matrix. Missing rows are filled with column means. This augmented feature matrix is passed to Stage 3. The rationale is that Tox21 endpoints are biologically correlated: a compound activating NR-AR is likely related to NR-AR-LBD (same protein, different assay format), and ER-active compounds often share scaffold features with AR-active ones. The meta-features allow the ensemble to exploit these cross-endpoint signals.

### Stage 3 — Endpoint-Aware Adaptive Ensemble

Rather than applying a single ensemble strategy uniformly, the pipeline uses an **endpoint-tier system** based on which algorithms performed best per endpoint in Stage 1:

#### Tier 1 — SPECIAL: `NR-AR`, `NR-AR-LBD`, `NR-ER` → CatBoost + BalancedRandomForest

These three endpoints are the most class-imbalanced androgen and estrogen receptor endpoints, where XGBoost and LightGBM alone did not yield adequate AUC. They require models that handle imbalance structurally:

- **CatBoost** with `auto_class_weights='Balanced'` — uses ordered boosting which reduces overfitting on small minority classes.
- **BalancedRandomForest (BRF)** from `imbalanced-learn` — trains each tree on a bootstrap with majority-class downsampling to achieve balanced splits. BRF is particularly robust for rare-event detection.

Ensemble: `0.35 × XGB + 0.35 × CatBoost + 0.30 × BRF`

#### Tier 2 — RF-STRONG: `NR-PPAR-gamma`, `SR-ATAD5`, `SR-HSE`, `NR-ER-LBD` → XGB + LightGBM + RF (RF-heavy)

For these endpoints, RandomForest contributed more than LightGBM in cross-validation. Ensemble weights are RF-heavy: `0.40 × XGB + 0.35 × RF + 0.25 × LightGBM`. RandomForest provides diversity through bagging and random feature subsets, with decorrelated trees that complement boosted models which tend to make correlated errors.

#### Tier 3 — RF-WEAK: `NR-AhR`, `NR-Aromatase`, `SR-ARE`, `SR-MMP`, `SR-p53` → XGB + LightGBM + RF (XGB-heavy)

For these endpoints, XGBoost and LightGBM dominate. Ensemble weights: `0.45 × XGB + 0.35 × LightGBM + 0.20 × RF`. LightGBM's leaf-wise tree growth finds better splits on high-dimensional feature sets, complementing XGBoost's depth-wise strategy.

All ensemble predictions are weighted averages of constituent model probability outputs across all 5 folds, maintaining the cross-validated estimation framework throughout.

### Stage 4 — Per-Endpoint Decision Threshold Optimisation

The optimal classification threshold per endpoint is computed using **Youden's J statistic**:

```
J = TPR - FPR  =  Sensitivity + Specificity - 1
```

The threshold maximising J on OOF predictions is saved per endpoint. Youden's J directly maximises the sum of sensitivity and specificity, improving recall over the default 0.5 threshold for imbalanced endpoints. The recall gain from threshold optimisation is reported per endpoint. Thresholds and recall gains are saved to `models/optimal_thresholds.json` and `results/optimal_thresholds.csv`.

### Stage 5 — Feature Importance Analysis

Feature importances are extracted from the XGBoost final models using **gain-based importance** (total loss improvement attributed to each feature across all trees). Importances are aggregated across all 12 endpoints to produce a global top-20 feature ranking and per-endpoint top-5 rankings. Feature types are labelled (ECFP fingerprint, Atom Pair, MACCS, physicochemical, SMARTS fragment, cross-endpoint OOF, 3D descriptor) for interpretability.

---

## Prediction Interface

A standalone `predict_smiles()` function is implemented that accepts any SMILES string and returns:

- Per-endpoint toxicity probabilities (0–1)
- Binary predictions using the optimised Youden J thresholds
- A formatted report of predicted toxic endpoints

The function applies the same VarianceThreshold mask, KNN imputation, and cross-endpoint meta-feature augmentation used during training, making it a faithful replica of the training-time inference pipeline.

Example usage (within the notebook after training):
```python
result = predict_smiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
```

---

## Results & Outputs

| Path | Contents |
|------|----------|
| `models/ensemble_models.pkl` | Trained ensemble model objects per endpoint |
| `models/xgb_models.pkl` | Trained XGBoost final models per endpoint |
| `models/cat_models.pkl` | Trained CatBoost models (special endpoints) |
| `models/brf_models.pkl` | Trained BalancedRandomForest models |
| `models/spw_dict.json` | Per-endpoint scale_pos_weight values |
| `models/optimal_thresholds.json` | Youden J optimal thresholds per endpoint |
| `results/ensemble_results.csv` | Per-endpoint XGB AUC, Ensemble AUC, and delta |
| `results/optimal_thresholds.csv` | Threshold, recall@0.5, recall@J, and recall gain |
| `results/feature_importances.csv` | Feature gain scores per endpoint |
| `features_raw.csv` | Full feature matrix (for reproducibility) |
| `plots/ensemble_results.png` | Bar chart: Ensemble vs XGBoost AUC per endpoint |
| `plots/roc_curves_ensemble.png` | ROC curves for all 12 endpoints |
| `plots/feature_importance_analysis.png` | Top feature importances visualisation |
| `plots/prediction_profile.png` | Example compound toxicity profile |

The ensemble consistently improves over the XGBoost baseline across most endpoints. Endpoints in the SPECIAL tier (NR-AR, NR-AR-LBD, NR-ER) showed the largest gains from CatBoost and BalancedRandomForest, which are specifically designed for the severe class imbalance those endpoints exhibit.

---

## Repository Structure

```
tox21-toxicity-prediction/
│
├── Deployment_pipeline.ipynb   # Main pipeline notebook (all stages)
├── tox21.csv                   # Raw dataset (SMILES + 12 toxicity labels)
├── codecuresl.csv              # Pre-computed descriptor columns
│
├── models/
│   ├── ensemble_models.pkl
│   ├── xgb_models.pkl
│   ├── cat_models.pkl
│   ├── brf_models.pkl
│   ├── spw_dict.json
│   └── optimal_thresholds.json
│
├── results/
│   ├── ensemble_results.csv
│   ├── optimal_thresholds.csv
│   └── feature_importances.csv
│
└── plots/
    ├── ensemble_results.png
    ├── roc_curves_ensemble.png
    ├── feature_importance_analysis.png
    └── prediction_profile.png
```

---

## Setup & Reproduction

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn scipy
```

RDKit is best installed via conda:
```bash
conda install -c conda-forge rdkit
```

### Running the Pipeline

1. Place `tox21.csv` (with SMILES column and all 12 toxicity label columns) in the working directory.
2. Open `Deployment_pipeline.ipynb` in Jupyter.
3. Run all cells in order. The notebook is fully self-contained — all feature matrices, models, and result files are generated within the execution.

### Note on the Uploaded Dataset

The uploaded `codecuresl.csv` contains pre-computed descriptor columns matching the pipeline output, minus Morgan fingerprint columns (omitted for upload size). To reproduce the full feature set, the full `tox21.csv` must be supplied and the pipeline run from Cell 1, which computes all 11 feature layers from SMILES strings via RDKit.

---

## Design Decisions & Justifications

**Why 11 feature layers?** No single molecular representation captures all relevant chemical information. Fingerprints encode structural patterns; physicochemical descriptors encode ADME-relevant properties; 3D descriptors encode shape; SMARTS patterns encode expert pharmacophore knowledge. The union of all layers maximises information available to the models, while VarianceThreshold and KNN imputation ensure the expanded feature space does not degrade model quality.

**Why not neural networks / GNNs?** Graph Neural Networks (ChemProp, AttentiveFP) achieve state-of-the-art on Tox21. However, tree-based ensembles on hand-crafted features are more interpretable (feature importances are directly actionable), faster to train without GPU requirements, and more robust on small per-endpoint labelled datasets where some endpoints have fewer than 500 labelled compounds. The adaptive ensemble strategy also allows endpoint-specific model selection without end-to-end retraining.

**Why CatBoost specifically for NR-AR/NR-ER?** Androgen and estrogen receptor agonism/antagonism is highly structure-specific, dominated by a small number of compound classes (steroids, phenols, bisphenols). The minority positive class is structurally coherent but very rare. CatBoost's ordered boosting reduces information leakage during training on small imbalanced datasets, and its native `auto_class_weights='Balanced'` avoids the manual `scale_pos_weight` tuning needed for XGBoost.

**Why BalancedRandomForest for SPECIAL endpoints?** BRF trains each tree on a bootstrap sample with majority-class downsampling to achieve class balance at the tree level. This is structurally different from sample reweighting: it changes the training distribution seen by each tree rather than adjusting the loss, making it complementary to CatBoost's reweighting approach and improving ensemble diversity.

**Why KNN imputation over mean/median?** Physicochemical descriptors are correlated — large molecules tend to have high HeavyAtoms, high MW, and often higher TPSA simultaneously. KNN imputation exploits this local correlation structure: it fills a missing descriptor value using the weighted average in the 5 nearest neighbours in descriptor space. Mean imputation would ignore this structure and introduce bias toward the global mean, particularly harmful for descriptors with bimodal or heavy-tailed distributions common in drug-like chemical space.

**Why Youden's J for threshold selection?** In toxicity prediction, both false negatives (missing a toxic compound) and false positives (flagging a safe compound as toxic) have real costs. Youden's J maximises sensitivity + specificity simultaneously, providing the threshold that best balances both error types. Per-endpoint threshold optimisation is critical because the optimal threshold varies substantially across endpoints due to differing class priors and score distributions.

**Why `scale_pos_weight` computed from EDA before training?** The class balance statistics (n_safe / n_toxic per endpoint) are computed in a dedicated EDA cell and persisted to `scale_pos_weights.json` before any model training begins. This ensures the weight is computed on the labelled-only subset of each endpoint (not the full dataset including unlabelled rows), which is the correct denominator.
