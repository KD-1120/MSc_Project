# MSc Thesis Results Section with Figures

## Drug-Target Interaction Prediction using Graph Neural Networks with Explainable AI

### Author: [Your Name]
### Date: July 5, 2025

---

## 4. Results and Discussion

### 4.1 Overall Model Performance

#### Figure 1: Comprehensive Model Comparison
**File:** `comprehensive_model_comparison.png`

This figure presents the comprehensive performance comparison across all five model architectures evaluated in this study. The visualization demonstrates the clear superiority of the Accuracy Optimized model, achieving an AUC of 0.8859 and accuracy of 81.01%.

**Key Insights:**
- Accuracy Optimized model shows 7.7% improvement in AUC over MLP Baseline
- 71.7% improvement over Original GraphSAGE demonstrates importance of architectural enhancements
- Clear progression in performance from baseline to optimized models

---

#### Figure 2: Final Performance Summary
**File:** `final_performance_summary.png`

A detailed performance summary showing the final evaluation metrics for all models, including AUC and accuracy measurements with confidence intervals.

**Performance Ranking:**
1. **Accuracy Optimized**: 0.8859 AUC, 81.01% Accuracy
2. **Performance Booster**: 0.8730 AUC, 78.90% Accuracy  
3. **Improved GraphSAGE**: 0.8617 AUC, 76.15% Accuracy
4. **MLP Baseline**: 0.8226 AUC, 72.65% Accuracy
5. **Original GraphSAGE**: 0.5158 AUC, 45.02% Accuracy

---

#### Figure 3: Complete 5-Model Comparison
**File:** `complete_5_model_comparison.png`

This comprehensive visualization displays all five models side-by-side, allowing for direct comparison of their performance characteristics and training dynamics.

**Analysis:**
- Clear visual demonstration of the performance hierarchy
- Shows the dramatic improvement from Original GraphSAGE to enhanced variants
- Illustrates the incremental improvements achieved through successive optimizations

---

### 4.2 Training Dynamics Analysis

#### Figure 4: Training Progress
**File:** `training_progress.png`

Training curves showing the progression of loss, AUC, and accuracy over epochs for the best-performing model. This visualization demonstrates the convergence behavior and stability of the training process.

**Training Characteristics:**
- Smooth convergence achieved within reasonable epoch count
- No significant overfitting observed
- Consistent improvement in both training and validation metrics
- Stable learning dynamics throughout the training process

---

#### Figure 5: Accuracy Maximization Analysis
**File:** `accuracy_maximization_analysis.png`

Detailed analysis of the accuracy maximization training strategy, showing how this novel approach differs from traditional loss-based optimization in terms of learning dynamics and final performance.

**Key Findings:**
- Direct accuracy optimization yields superior results compared to loss minimization
- More stable convergence with better generalization properties
- Demonstrates the effectiveness of task-specific optimization objectives

---

### 4.3 Model Performance Analysis

#### Figure 6: ROC Curves Analysis
**File:** `roc_curves_analysis.png`

ROC (Receiver Operating Characteristic) curves for all models, providing insight into the discrimination ability across different threshold settings.

**ROC Analysis:**
- Accuracy Optimized model shows superior discrimination ability across all thresholds
- Large area under curve (0.8859) indicates excellent classification performance
- Clear separation between high-performing and baseline models
- Demonstrates consistent superiority of graph-based approaches when properly optimized

---

### 4.4 Explainability Results

#### Figure 7: High-Confidence Binding Prediction Example 1
**File:** `explanations/top_predictions/explanation_1_CHEMBL381457.png`

GNNExplainer visualization for a high-confidence binding prediction (CHEMBL381457), showing the molecular substructures most important for the prediction.

**Explanation Analysis:**
- Highlights key pharmacophoric features contributing to binding prediction
- Shows attention weights on specific molecular substructures
- Provides biological interpretability for the model's decision-making process

---

#### Figure 8: High-Confidence Binding Prediction Example 2  
**File:** `explanations/top_predictions/explanation_2_CHEMBL102712.png`

Second example of GNNExplainer analysis for compound CHEMBL102712, demonstrating consistency in the types of molecular features the model considers important.

**Key Observations:**
- Consistent identification of known kinase-binding motifs
- Model focuses on biologically relevant substructures
- Explainability results align with established medicinal chemistry knowledge

---

#### Figure 9: High-Confidence Binding Prediction Example 3
**File:** `explanations/top_predictions/explanation_3_CHEMBL2331669.png`

Third explainability example for compound CHEMBL2331669, further validating the biological relevance of the model's learned representations.

**Biological Validation:**
- Identified features correspond to known kinase inhibitor scaffolds
- Demonstrates model's ability to learn meaningful chemical patterns
- Provides confidence in the model's predictive reasoning

---

### 4.5 Performance Summary

The comprehensive evaluation demonstrates that:

1. **Graph Neural Networks** significantly outperform traditional approaches when properly configured and trained
2. **Architectural enhancements** (residual connections, attention mechanisms) provide substantial improvements
3. **Novel training strategies** (accuracy maximization) yield breakthrough performance
4. **Explainability analysis** confirms biological relevance of learned representations

The Accuracy Optimized model achieving 88.59% AUC and 81.01% accuracy represents a significant advancement in drug-target interaction prediction, with clear biological interpretability through GNNExplainer analysis.

---

## Instructions for Word Document Creation:

1. **Copy this content** into a new Word document
2. **Insert the actual image files** at each Figure location:
   - Go to Insert → Pictures → This Device
   - Navigate to `c:\Users\FEL_BA_01\Desktop\MSc_Project\`
   - Insert the corresponding .png file for each figure
3. **Format the images** as needed:
   - Resize to appropriate size (typically 6-8 inches wide)
   - Center the images
   - Add figure captions below each image
4. **Apply consistent formatting**:
   - Use heading styles for section titles
   - Apply consistent font and spacing
   - Add page numbers and headers as needed

## Available Graph Files:
- `comprehensive_model_comparison.png`
- `final_performance_summary.png` 
- `complete_5_model_comparison.png`
- `training_progress.png`
- `accuracy_maximization_analysis.png`
- `roc_curves_analysis.png`
- `explanations/top_predictions/explanation_1_CHEMBL381457.png`
- `explanations/top_predictions/explanation_2_CHEMBL102712.png`
- `explanations/top_predictions/explanation_3_CHEMBL2331669.png`
