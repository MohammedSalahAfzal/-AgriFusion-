# Agrifusion Net: Hybrid EfficientNetV2 and Transformer for Rice Leaf Disease Classification with Farmer Advisory Support

## Abstract
Agrifusion Net is a computer-vision prototype designed to classify rice leaf images into disease/healthy categories and present cautious next-step guidance. The feature extractor is EfficientNetV2-S; its spatial feature map is projected into a sequence of tokens and processed by a Transformer encoder before classification. A Flask web interface accepts a leaf image and displays the predicted class, model score, alternative class scores, and rule-based advisory text. The project is intended for education and research. It is not a replacement for agricultural diagnosis or extension services.

## 1. Problem statement
Rice diseases can reduce crop health and yield. Visual assessment requires experience and can be difficult when symptoms look similar. This project explores whether a deep-learning image classifier can support preliminary screening from photographs and make general information easier to access.

## 2. Aim and objectives
**Aim:** Build a hybrid deep-learning system for rice leaf disease image classification and advisory support.

Objectives:
- Prepare a labelled image dataset and separate it into training, validation, and test subsets.
- Train a hybrid EfficientNetV2-S + Transformer model.
- Evaluate performance with accuracy, per-class precision/recall/F1, macro F1, and a confusion matrix.
- Deploy a simple Flask interface for image upload and result display.
- Provide cautious, editable guidance that encourages expert confirmation.

## 3. Scope
The project classifies only the labels represented in the training dataset. It does not identify every possible rice disease, quantify disease severity, recommend a chemical dose, or confirm disease from field symptoms. The included advisory is a transparent rule-based mapping, not a BioBERT or generative-language model.

## 4. Proposed methodology
1. **Dataset collection:** Select a licensed rice-leaf image dataset and document the source and label definitions.
2. **Data split:** Use independent train, validation, and test sets. Prevent duplicate/near-duplicate images and plant-level leakage across splits.
3. **Preprocessing:** Resize RGB images to 224 × 224 pixels and normalize using ImageNet channel statistics.
4. **Training augmentation:** Apply random horizontal flips, small rotations, and mild color jitter to training images only.
5. **Feature extraction:** EfficientNetV2-S produces a spatial feature map.
6. **Token projection:** A 1 × 1 convolution maps feature channels to a 256-dimensional embedding. The spatial positions become tokens and receive learnable positional embeddings.
7. **Transformer encoder:** Two encoder layers model relationships between spatial tokens.
8. **Classification:** Mean pooling, layer normalization, dropout, and a linear head produce class logits. Cross-entropy loss and AdamW optimize model parameters.
9. **Model selection:** Save the checkpoint with the best validation accuracy; keep the test split untouched during this choice.
10. **Evaluation and deployment:** Calculate test metrics and serve the checkpoint through Flask.

## 5. System requirements
Python 3.10/3.11, PyTorch, torchvision, Flask, Pillow, NumPy, and scikit-learn. A CUDA-compatible GPU is optional; CPU execution is supported but can be slow. ImageNet pretrained weights are optional and require a first-run download.

## 6. Evaluation plan
Report the following after running `evaluate.py`:
- Dataset name, URL/citation, license, class labels, and class counts.
- Train/validation/test split sizes and split methodology.
- Accuracy and macro F1 on the held-out test set.
- Per-class precision, recall, and F1-score.
- Confusion matrix and examples of incorrect predictions.
- Hardware, training duration, epoch count, image size, and whether pretrained weights were used.
- Limitations and a comparison against a simple baseline, if conducted.

**Results:** Not yet measured. Populate this section from `artifacts/evaluation.json` after training. Do not insert guessed values.

| Metric | Result |
|---|---|
| Test images | Fill from evaluation output |
| Test accuracy | Fill from evaluation output |
| Macro F1 | Fill from evaluation output |
| Per-class precision/recall/F1 | Fill from evaluation output |

## 7. Application design
The Flask application validates that an image can be opened, applies the same resize and normalization as evaluation, obtains class scores, and displays the top result. The selected label is mapped to advisory text in `advisory.json`. The interface explicitly describes the output as screening, not confirmation.

## 8. Ethical and safety considerations
Incorrect classifications may lead to unnecessary cost or delayed intervention. Confidence scores are not calibrated probabilities. The model may perform poorly on field images unlike the training data, including different cultivars, lighting, camera types, disease stages, and backgrounds. Users should seek local agricultural expert confirmation and follow local regulations. Dataset terms must be respected; images should not be collected or shared without appropriate permission.

## 9. Limitations and future work
Potential improvements include collecting diverse field images, external validation across locations/seasons, class-imbalance handling, calibration, explainability (e.g., Grad-CAM), benchmarking against EfficientNetV2-only and CNN-only baselines, uncertainty-based abstention, multilingual advisory content reviewed by agronomists, and a carefully evaluated offline/mobile deployment. A text model should only be added if there is a clearly defined, sourced advisory corpus and a separate evaluation plan.

## 10. Conclusion
Agrifusion Net provides an end-to-end framework linking a hybrid image classifier with a simple web interface. Its actual value must be established through reproducible training and held-out evaluation. No performance result is asserted before those experiments are completed.

## References to cite in the final submission
- Tan, M. and Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training.* International Conference on Machine Learning (ICML).
- Vaswani, A. et al. (2017). *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS).
- The exact rice-leaf dataset page and license used for the experiments.
- Official PyTorch and torchvision documentation for the implementation.
