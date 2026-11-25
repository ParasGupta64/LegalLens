# LegalLens
### AI-Powered Legal Text Analysis and Document Understanding

LegalLens is an academic project that explores how artificial intelligence and natural language processing can make legal information easier to analyze and understand.

The application brings together legal sentence comparison, legal-language simplification, document-based question answering, and OCR in a single Streamlit interface.

## Features

- **Legal meaning comparison** — assess whether two legal sentences preserve the same meaning.
- **Legal text simplification** — rewrite complex legal wording in more accessible language.
- **Document question answering** — upload a document and ask questions about its contents.
- **OCR-based text extraction** — extract text from supported document images.
- **AI-assisted explanations** — provide accessible explanations to support the user's understanding of legal text.

## Technology

- **Language:** Python
- **Application:** Streamlit
- **Machine learning:** PyTorch, Transformers, Sentence Transformers
- **Language processing:** NLP and text-processing tools
- **Document and image processing:** PDF/DOCX parsing and OCR
- **Generative AI:** OpenAI API

## Run locally

Python 3.12 is recommended. Run commands from the repository root.

```bash
git lfs install
git clone https://github.com/ParasGupta64/LegalLens.git
cd LegalLens
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` and use `copy .env.example .env`.

Set `OPENAI_API_KEY` in `.env` to enable external generation. Without a key, simplification uses local substitutions and chat displays matching excerpts. Relevant document excerpts are sent to the external provider when generation is enabled. Never commit real API keys.

The preservation model requires local Git LFS weights and may download its encoder from Hugging Face on first use. A Git LFS pointer is not a usable model file. For image OCR, install the Tesseract executable as well as the Python package, with `eng` and `hin` language data for English and Hindi; Windows native OCR is also attempted. Text extraction from scanned PDFs does not automatically invoke image OCR.

## Repository index and paper

See [INDEX.md](INDEX.md) for the code/data map and research limitations.

- [IEEE paper (PDF)](paper/LegalLens_IEEE_Paper.pdf)
- [Editable paper (Word)](paper/LegalLens_IEEE_Paper.docx)

## Validation

```bash
python -m unittest discover -s tests -v
```

These offline tests cover rendering safety, upload limits, retrieval, chunking, and grouped data preparation. They do not certify legal accuracy or call paid APIs.

## Multilingual experiments

```bash
python article_src/dataset_creation/generate_multilingual_judge.py
python article_src/src/prepare_multilingual_training_corpora.py
python article_src/src/run_all_experiments.py
```

The last command trains models and can take substantial time; it overwrites the two checkpoint directories. Back up existing weights first. Generated datasets and predictions stay ignored by default. The benchmark is programmatically constructed, and source groups are kept within one split to prevent exact original-clause leakage. Cross-document and translation-level independence remain unverified.

The supplied checkpoint files share one Git LFS object. No empirical superiority claim follows from their directory names. Archived generators contain unverified fixed values and are disabled. Use measured experiment outputs for any future performance claims.

The inherited research scripts under `article_src/` have their own dependency files; they are separate from the application installation.

## Research and Third-Party Attribution

LegalLens is the project application. The repository also contains material associated with the JUDGEBERT research and dataset. That material remains attributable to its original authors and is subject to its applicable license.

The JUDGEBERT dataset is identified as licensed under **CC BY-NC-SA 4.0**. Review the [license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) before using or redistributing the dataset or adapted material.

If you use the JUDGEBERT research, cite the [original paper](https://aclanthology.org/2025.emnlp-main.5/) and retain [original repository attribution](https://github.com/GRAAL-Research/JUDGEBERT):

```bibtex
@inproceedings{beauchemin2025judgebert,
  title={JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences},
  author={Beauchemin, David and Albert-Rochette, Michelle and Khoury, Richard and D{\\'e}ziel, Pierre-Luc},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={92--118},
  year={2025}
}
```

## Team

| Member | Student ID |
| --- | --- |
| Paras Gupta | S24CSEU1954 |
| Harsh Goyal | S24CSEU1936 |
| Harshit Khatri | S24CSEU1951 |
| Daksh Pratap Singh | S24CSEU2285 |

Instructor: Surabhi Solanki.

## Project Maintainer

**Paras Gupta**  
B.Tech, Computer Science and Engineering (AI & ML)  
Bennett University, Greater Noida, India

[GitHub Profile](https://github.com/ParasGupta64)

---

*LegalLens is an academic project. Its outputs are intended to assist with understanding legal text and should not be treated as professional legal advice.*
