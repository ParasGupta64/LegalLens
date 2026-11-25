# LegalLens

### Assessing Legal Meaning Preservation Between Sentences

LegalLens is an academic legal NLP application for comparing an original clause with a proposed simplification. It predicts a preservation score, a three-way decision, and diagnostic labels for possible changes to obligations, conditions, exceptions, and other legal details. The Streamlit interface supports English and Hindi, document chat, and image text extraction.

[Research paper (PDF)](paper/LegalLens_IEEE_Paper.pdf) · [Editable paper (Word)](paper/LegalLens_IEEE_Paper.docx) · [Repository index](INDEX.md)

## Features

| Feature | Behavior |
| --- | --- |
| Legal meaning comparison | Score a clause pair from 1 to 10 with an optional context window |
| Preservation decisions | Predict Preserved, Partially Preserved, or Not Preserved |
| Diagnostic explanations | Describe predicted changes to legal content using nine error categories |
| Clause simplification | Generate an English or Hindi rewrite, with local substitutions as a fallback |
| Document chat | Retrieve passages from PDF/DOCX files and optionally generate an answer |
| Image OCR | Extract text using compatible Windows OCR engines or Tesseract |

Predictions require review against the original document. The supplied model can miss material obligation changes; it is not a validated legal decision system.

## Quick start

### Prerequisites

- Python 3.12 is recommended.
- Git and **Git LFS** must be installed to retrieve the model weights.
- Internet access may be needed to download the pretrained encoder on first use.
- Image OCR requires the Tesseract executable and language data (`eng` for English, `hin` for Hindi), unless a compatible Windows OCR engine is available.

### 1. Clone and retrieve weights

```bash
git lfs install
git clone https://github.com/ParasGupta64/LegalLens.git
cd LegalLens
git lfs pull
```

The two checkpoint directories each contain a large model file. A small Git LFS pointer file is not usable model weights.

### 2. Create an environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### 3. Configure optional generation

Set the following in `.env` to enable OpenAI-powered simplification and document answers:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Without a key, simplification uses local substitutions and document chat displays matching excerpts. Model-based sentence comparison still requires the checkpoint and encoder. When external generation is enabled, relevant input text is sent to the provider. Keep real keys out of Git.

### 4. Launch

Run from the repository root:

```bash
streamlit run app.py
```

For a command-line model demonstration:

```bash
python demo.py --interactive
```

## Usage

1. Open **Legal Evaluator** and select a sample or enter an original clause and candidate rewrite.
2. Choose English or Hindi and supply relevant context when available.
3. Evaluate the pair, then inspect the score, decision, and diagnostic output against the source.
4. Open **AI Document Chat** to upload a PDF or DOCX file (maximum 20 MB) and ask about its contents.

Image OCR is a separate workflow. Uploading a scanned PDF does not automatically invoke OCR. Context windows and token limits can omit relevant definitions or exceptions.

## Research paper and reported results

The [six-page IEEE-format paper](paper/LegalLens_IEEE_Paper.pdf) includes eight tables covering the corpus, model settings, baseline/context/language comparisons, sanity checks, and fifteen selected cases.

The following figures reproduce the supplied project report. They have **not been independently reproduced** from a complete set of run artifacts in this repository.

| Model / split | Pearson r | RMSE | Decision accuracy |
| --- | ---: | ---: | ---: |
| Generic SBERT (cosine similarity) | 0.2959 | 3.2125 | N/A |
| LegalLens sentence-only | 0.6511 | 3.1017 | 51.0% |
| LegalLens context-aware | 0.7676 | 2.3375 | 53.9% |
| English sub-corpus | 0.7470 | 2.3124 | 47.1% |
| Hindi sub-corpus | 0.7888 | 2.3623 | 60.8% |

The report also gives MAE 0.33 and 15/15 decision agreement on selected cases. That selected-case agreement is not generalization accuracy. The shipped sentence-only and context-aware weights reference the same Git LFS object; their directory names do not establish two separately trained models. See [evidence and limitations](INDEX.md#evidence-and-limitations) before interpreting the results.

## Repository map

```text
LegalLens/
├── app.py                 # Streamlit application
├── demo.py                # Command-line evaluator
├── article_src/           # Model, preprocessing, retrieval, and research code
├── checkpoints/           # Git LFS weights and tokenizers
├── datastore/             # Research datasets and generated-data destinations
├── annotation_tool/       # Original annotation tooling
├── paper/                 # Research paper in Word and PDF
├── tests/                 # Offline regression and UI smoke tests
├── archive/               # Disabled generators with unverified fixed results
├── .env.example           # Optional generation configuration
└── requirements.txt       # Application dependencies
```

See [INDEX.md](INDEX.md) for individual modules and their roles.

## Validation

```bash
python -m unittest discover -s tests -v
```

Nine tests cover HTML escaping, file validation, chunking, retrieval abstention, source-grouped splits, checkpoint-pointer errors, explanations, and application navigation. UI tests mock inference; they do not validate the model's legal accuracy or call paid APIs.

## Reproducing experiments

```bash
python article_src/dataset_creation/generate_multilingual_judge.py
python article_src/src/prepare_multilingual_training_corpora.py
python article_src/src/run_all_experiments.py
```

The final command trains models and overwrites both checkpoint directories. Back up existing weights before running it. Generated datasets and predictions are ignored by default.

The generator produces 320 records but only 120 exact original/candidate pairs, with programmatically supplied labels. Current splits group original clauses to prevent exact source-text leakage. Document-level and translation-level independence still require provenance identifiers. Retain split membership, checkpoint hashes, and per-example predictions for future result verification.

The inherited research tools have separate dependency files under `article_src/`; the root requirements are for the application and multilingual workflow. Archived report generators contain unverified fixed figures and are disabled.

## Technology

Python · Streamlit · PyTorch · Hugging Face Transformers · Sentence Transformers · scikit-learn TF-IDF retrieval · pypdf · python-docx · Pillow · Tesseract / Windows OCR · optional OpenAI API

## Attribution and licensing

LegalLens includes material from the original [JUDGEBERT research](https://aclanthology.org/2025.emnlp-main.5/) and [repository](https://github.com/GRAAL-Research/JUDGEBERT). That work remains attributable to its original authors. The dataset is identified as licensed under **CC BY-NC-SA 4.0**; review [LICENSE](LICENSE) and the [license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/) before reuse or redistribution.

Original research citation:

> “JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences,” Proceedings of EMNLP, 2025, pp. 92–118. DOI: [10.18653/v1/2025.emnlp-main.5](https://doi.org/10.18653/v1/2025.emnlp-main.5).

LegalLens supports academic exploration and document understanding. Its outputs are not a substitute for professional legal advice.
