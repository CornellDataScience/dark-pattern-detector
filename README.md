# Dark Pattern Detector

A research project for identifying potentially manipulative interface patterns in website and app screenshots using visual, textual, and structural evidence.

Based on [Misleading Patterns — CDS Onboarding](https://docs.google.com/document/d/1w7mKr95aSBic2trthV9Wxaq-8utEPkl5tnn_mZmZJj8/edit?tab=t.0), including its model-design diagrams.

## Status and scope

**Documentation only.** The repository currently contains this project outline and a `.gitignore`. No parsing pipeline, models, datasets, application, or evaluation results are included. There are no installation or run commands yet.

The current scope is repository setup and documentation of the architecture and research plan. All components below are proposed; implementation choices remain open.

## Research objective

Compare text-only, vision-only, UI-structure, and multimodal approaches to determine which signals help identify dark patterns. The intended output is a set of pattern predictions with scores and supporting UI elements or regions.

The onboarding document also proposes a smaller click-to-cancel MVP: parse an interface with OmniParser and use a model to assess cancellation-related UI. Verifying that cancellation actually succeeds would require interaction evidence beyond a single screenshot. The MVP and the broader research pipeline are separate possible implementation scopes.

## Detection categories

| Candidate category | Example evidence | Interpretation limit |
| --- | --- | --- |
| False urgency | Countdown timers or urgency language | A screenshot can reveal urgency cues, but cannot establish that the urgency is false. |
| Visual interference | A prominent Accept button and a low-visibility Reject button | Assess relative size, contrast, and placement in context. |
| Hidden costs | Fee disclosures in small or low-visibility text | Additional screens or pricing context may be needed to establish that a cost was hidden. |
| Confirmshaming | Emotionally manipulative refusal wording | Retain the exact text and its surrounding choice. |
| Sneaking / preselection | A paid optional add-on already selected | Selection state and pricing context are relevant evidence. |

Multiple categories may apply to one screenshot. An **insufficient evidence** outcome should be distinct from a negative prediction. The model-design diagram uses related labels such as urgency/scarcity cues and low-visibility disclosures; the final taxonomy must reconcile these observable cues with the broader categories above.

## Proposed architecture

The design combines parsed UI elements with a full-image visual branch that preserves information the parser may miss.

```mermaid
flowchart TD
    A[UI screenshot] --> B[OmniParser and OCR]
    A --> C[Full-image visual encoder]
    B --> D[Element crops, text, roles, and bounding boxes]
    D --> E[Visual, text, and structural encoders]
    E --> F[Element-level multimodal fusion]
    F --> G[Relational transformer]
    C --> G
    G --> H[Multi-label classification]
    G --> I[Evidence localization]
```

| Component | Responsibility | Candidates in the onboarding document |
| --- | --- | --- |
| Input processing | Load screenshots and optional HTML | Pillow, BeautifulSoup |
| UI parsing | Extract text, UI regions, bounding boxes, and element roles | OmniParser V2, AidUI UIED, OCR |
| Visual encoding | Represent the screenshot and individual element crops | CLIP / ViT, ResNet |
| Text encoding | Represent UI wording | MiniLM, DeBERTa-v3-small, BERT, RoBERTa |
| Structural encoding | Represent geometry, relative prominence, contrast, and roles | OpenCV, bounding-box features, MLP |
| Fusion | Combine visual, text, and structural representations | Concatenation with an MLP; relational or cross-modal attention |
| Prediction | Produce pattern scores and supporting regions | Multi-label sigmoid classifier, element-localization head |
| Results interface | Present predictions and evidence | React with FastAPI, if a demo is included in a later scope |

For element `i`, the proposed fusion is `z_i = MLP([v_i; t_i; s_i])`, where `v_i`, `t_i`, and `s_i` are its visual, textual, and structural embeddings. The model-design diagrams propose a shared 256-dimensional representation and a relational transformer with two layers and eight attention heads.

The source contains two related transformer designs: one combines element tokens with full-image patch tokens; the other uses an Element-Aware Multimodal Transformer (EAMT) with a global screenshot token and a maximum of 64 UI elements. These are design candidates, not a finalized implementation. The choice of global representation, element-selection policy, and localization supervision remains open.

## Research plan

### 1. Prepare and standardize data

Evaluate the document's candidate sources: AidUI / ContextDP, Dark Patterns at Scale, and custom labeled screenshots. Check the available modalities and annotations before selecting a source for an experiment.

Define a shared representation for screenshot identity, source domain, extracted text, UI elements, bounding boxes, and category labels. Record which samples have localization annotations and which require additional context. Reconcile label differences before combining datasets.

### 2. Establish evaluation splits

Create train, validation, and test splits separated by website or app source to reduce leakage. Keep duplicate or closely related screenshots together. Use the validation split for model selection and reserve the test split for final comparisons.

### 3. Implement controlled baselines

| Experiment | Input | Research question |
| --- | --- | --- |
| NLP-only | Extracted UI text | How much can wording explain? |
| CV-only | Raw screenshot | How much can appearance explain? |
| UI structure | Bounding boxes and geometric features | How much can layout explain? |
| Pretrained VLM | Screenshot with text | How well does pretrained multimodal reasoning perform? |
| Learned multimodal model | Visual, textual, and structural features | Does learned fusion improve detection and localization? |

Use consistent labels and data splits across comparisons. Record differences in available input, pretraining, and supervision so results can be interpreted fairly.

### 4. Train and evaluate fusion models

Compare simple MLP fusion with the proposed relational transformer. Evaluate macro-F1, precision, recall, AUROC, and per-category F1. Add a localization metric when bounding-box or element-level annotations support it; choose the metric to match those annotations.

Run ablations that remove the visual, textual, and structural branches individually. Compare the contribution of full-image features with parsed element features, and inspect errors caused by OCR or UI parsing.

### 5. Report findings

Report performance by category, ablation results, and representative errors with supporting screenshots or UI regions. Document dataset versions, splits, preprocessing, model configurations, random seeds, and decision thresholds alongside each experiment. Report insufficient-evidence outcomes separately so they are not silently counted as clean interfaces.

## Decisions before implementation

- Choose the first deliverable: cancellation-focused MVP, a baseline experiment, or the broader research pipeline.
- Select datasets and reconcile category labels, negative examples, and insufficient-evidence annotations.
- Confirm available compute and whether models run locally or use a hosted API.
- Select a fusion architecture and the supervision available for localization.
- Decide whether the deliverable needs a CLI, notebooks, an API, or an interactive demo.

## References

- [Project onboarding and model design](https://docs.google.com/document/d/1w7mKr95aSBic2trthV9Wxaq-8utEPkl5tnn_mZmZJj8/edit?tab=t.0)
- [AidUI](https://github.com/SageSELab/AidUI)
- [Microsoft OmniParser](https://github.com/microsoft/OmniParser)
- [Dark Patterns at Scale data](https://github.com/aruneshmathur/dark-patterns/tree/master/data)

These are references from the onboarding document; no external code, model weights, or datasets have been incorporated into this repository.
