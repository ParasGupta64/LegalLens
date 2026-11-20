"""
demo.py - Live Model Inference & Interactive CLI Demo for LegalLens

Run this anytime using:
    python demo.py
    python demo.py --interactive
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure src modules are discovered
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "article_src", "src"))

import torch
from multilingual_judgebert import load_checkpoint, predict_single_pair


def display_result(pair_id, lang, domain, context, original, simplified, pred):
    print("\n" + "=" * 75)
    print(f"CASE: {pair_id} | Language: {lang.upper()} | Domain: {domain}")
    print("=" * 75)
    if context:
        print(f"Context:    {context}")
    print(f"Original:   {original}")
    print(f"Simplified: {simplified}")
    print("-" * 75)
    print("LIVE MODEL OUTPUT (Multilingual JUDGEBERT):")
    print(f"  * LMP Score:           {pred['lmp_score']:.2f} / 10.0")
    print(f"  * Decision:            {pred['decision']}")
    errs = pred["error_types"]
    print(f"  * Legal Error Type:    {', '.join(errs) if errs != ['none'] else 'None (Accurate)'}")
    print(f"  * Explanation:         {pred['explanation']}")
    print("=" * 75)


def run_interactive(model, tokenizer, mode, device):
    print("\n--- Interactive Legal Meaning Preservation Evaluator ---")
    print("Type 'exit' or press Ctrl+C at any prompt to quit.\n")
    idx = 1
    while True:
        try:
            lang = input("Language [en/hi] (default: en): ").strip().lower() or "en"
            if lang == "exit":
                break
            context = input("Document/Clause Context (optional, press Enter to skip): ").strip()
            if context.lower() == "exit":
                break
            original = input("Original Legal Sentence: ").strip()
            if not original or original.lower() == "exit":
                break
            simplified = input("Simplified Candidate Sentence: ").strip()
            if not simplified or simplified.lower() == "exit":
                break

            pred = predict_single_pair(
                model,
                tokenizer,
                original=original,
                simplification=simplified,
                context=context,
                lang=lang if lang in ("en", "hi") else "en",
                mode=mode if context else "sentence_only",
                device=device,
            )
            display_result(f"CUSTOM_{idx:02d}", lang, "Custom Input", context, original, simplified, pred)
            idx += 1
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Live JUDGEBERT Multilingual Demo")
    parser.add_argument("--interactive", action="store_true", help="Jump straight to interactive input loop.")
    ckpt_default = os.path.join("checkpoints", "context_aware")
    if not os.path.exists(os.path.join(ckpt_default, "model.pt")):
        ckpt_default = os.path.join("checkpoints", "sentence_only")
    parser.add_argument("--checkpoint", type=str, default=ckpt_default)
    args = parser.parse_args()

    print("\n" + "#" * 75)
    print("   ⚖️  LegalLens")
    print("#" * 75)

    if not os.path.exists(os.path.join(args.checkpoint, "model.pt")):
        print(f"\n[ERROR] Saved model checkpoint not found at '{args.checkpoint}'.")
        print("Run training first with:")
        print("    python article_src/src/run_all_experiments.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained checkpoint from '{args.checkpoint}' on {device}...")
    model, tokenizer, mode = load_checkpoint(args.checkpoint, device=device)
    print(f"Model loaded successfully (mode: {mode}).")

    if not args.interactive:
        samples = [
            {
                "pair_id": "EN_DEMO_01 (Obligation Shift)",
                "lang": "en",
                "domain": "Commercial Contracts",
                "context": "CONSULTING AGREEMENT - Clause 4.3: Payment Obligations",
                "original": "Client shall pay all undisputed invoice amounts within thirty (30) days of receipt.",
                "simplified": "The client may choose to pay invoice amounts whenever convenient within 30 days.",
            },
            {
                "pair_id": "HI_DEMO_02 (Preserved Statute)",
                "lang": "hi",
                "domain": "Statutory Acts",
                "context": "उपभोक्ता संरक्षण अधिनियम, 2019 - धारा 2(7): उपभोक्ता की परिभाषा",
                "original": "उपभोक्ता से ऐसा कोई व्यक्ति अभिप्रेत है जो किसी ऐसे प्रतिफल के लिए माल खरीदता है, जिसका संदाय किया जा चुका है या संदाय करने का वचन दिया गया है।",
                "simplified": "उपभोक्ता वह व्यक्ति है जो पैसे देकर या बाद में पैसे देने का वादा करके कोई सामान खरीदता है।",
            },
            {
                "pair_id": "EN_DEMO_03 (Exception Reversal)",
                "lang": "en",
                "domain": "Commercial Contracts",
                "context": "INDEMNITY AGREEMENT - Clause 8.4: Exclusions",
                "original": "Insurer shall not be liable for any damage caused by flood, earthquake, or war.",
                "simplified": "The insurance company will cover damage resulting from floods and earthquakes during wartime.",
            },
        ]

        for s in samples:
            pred = predict_single_pair(
                model,
                tokenizer,
                original=s["original"],
                simplification=s["simplified"],
                context=s["context"],
                lang=s["lang"],
                mode=mode,
                device=device,
            )
            display_result(
                s["pair_id"], s["lang"], s["domain"],
                s["context"], s["original"], s["simplified"], pred
            )

        try:
            ans = input("\nWould you like to test your own English/Hindi legal pair? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                run_interactive(model, tokenizer, mode, device)
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        run_interactive(model, tokenizer, mode, device)

    print("\nSession finished!")


if __name__ == "__main__":
    main()
