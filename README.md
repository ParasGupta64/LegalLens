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

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ParasGupta64/LegalLens.git
cd LegalLens
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root and add the API key required by your local configuration:

```env
OPENAI_API_KEY=your_api_key_here
```

Keep your API key private. Do not commit the `.env` file to GitHub.

### 4. Launch LegalLens

```bash
streamlit run app.py
```

## Repository Layout

```text
LegalLens/
├── app.py
├── article_src/
│   └── src/
├── datastore/
├── requirements.txt
└── README.md
```

*This is a high-level overview; additional project files and resources may be present in the repository.*

## Research and Third-Party Attribution

LegalLens is the project application. The repository also contains material associated with the JUDGEBERT research and dataset. That material remains attributable to its original authors and is subject to its applicable license.

The JUDGEBERT dataset is identified as licensed under **CC BY-NC-SA 4.0**. Review the [license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) before using or redistributing the dataset or adapted material.

If you use the JUDGEBERT research, cite the original work:

```bibtex
@inproceedings{beauchemin2025judgebert,
  title={JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences},
  author={Beauchemin, David and Albert-Rochette, Michelle and Khoury, Richard and D{\\'e}ziel, Pierre-Luc},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={92--118},
  year={2025}
}
```

## Project Maintainer

**Paras Gupta**  
B.Tech, Computer Science and Engineering (AI & ML)  
Bennett University, Greater Noida, India

[GitHub Profile](https://github.com/ParasGupta64)

---

*LegalLens is an academic project. Its outputs are intended to assist with understanding legal text and should not be treated as professional legal advice.*
