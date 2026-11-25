# Repository index

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit application: clause comparison, OCR, simplification, and document chat |
| `demo.py` | Command-line preservation-model demonstration |
| `article_src/src/multilingual_judgebert.py` | Multilingual multi-task model, training, evaluation, checkpoint loading |
| `article_src/src/context_preprocessor.py` | Pair/context formatting, decision thresholds, diagnostic taxonomy |
| `article_src/src/legal_simplifier.py` | External and deterministic simplification, image OCR, clause parsing |
| `article_src/src/doc_chat.py` | File validation/extraction, chunking, TF-IDF retrieval, optional generated answers |
| `article_src/src/ui_formatting.py` | HTML-safe chat formatting |
| `article_src/dataset_creation/generate_multilingual_judge.py` | Programmatically constructed English/Hindi examples |
| `article_src/src/prepare_multilingual_training_corpora.py` | Source-grouped splits and sanity-pair augmentation |
| `article_src/src/run_all_experiments.py` | Actual training and five evaluation experiments |
| `article_src/` (other scripts) | Original JUDGEBERT research tooling; separate optional dependencies |
| `datastore/` | Original research data and generated-data destinations |
| `checkpoints/` | Git LFS model weights and tokenizers |
| `annotation_tool/` | Original annotation tooling and instructions |
| `figs/` | Original research figures |
| `paper/` | Reviewed IEEE paper in Word and PDF |
| `tests/test_core.py` | Offline application/data regression tests |
| `archive/` | Disabled generators with fixed, unverified research results |
| `requirements.txt` | Application and multilingual experiment dependencies |
| `.env.example` | Optional external-generation configuration |

## Evidence and limitations

The dataset generator produces 320 records from 90 base records, with 120 exact original/candidate pairs. Labels are programmatic; independent legal annotation is not established. Splits now keep all variants of each normalized original clause together. Source-document and translated-example grouping need provenance identifiers that the current data does not supply.

A live smoke test on the supplied checkpoint returned 10/10 and Preserved for “The tenant must pay within 30 days.” versus “The tenant may pay within 30 days.” This is a material obligation change. Treat outputs as unvalidated model predictions; code cleanup does not repair or validate the weights.

The checked-in sentence-only and context-aware model files point to the same Git LFS object. Their names do not prove that two models were trained. Run the experiments with grouped splits to obtain separately trained checkpoints; previous paper figures must not be treated as verified results.

Original JUDGEBERT material remains attributable to its authors and retains its applicable licensing. See `README.md` and `LICENSE`.
