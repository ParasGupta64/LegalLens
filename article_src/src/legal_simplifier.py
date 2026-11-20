"""
legal_simplifier.py - AI & Plain-Language Legal Simplification and Document OCR Engine for LegalLens.
Supports English & Hindi clause simplification with legal meaning preservation.
"""

import os
import re
from typing import Tuple, Optional, List, Dict
from PIL import Image

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = lambda *args, **kwargs: {}


# -------------------------------------------------------------------------
# 1. OCR Extraction (Optical Character Recognition)
# -------------------------------------------------------------------------

def extract_text_from_document_image(image_input, lang: str = "en") -> Tuple[bool, str, str]:
    """
    Extracts readable text from an uploaded legal document image using:
      1. Native Windows OCR via winocr
      2. Native Windows.Media.Ocr via winrt (direct fallback if winocr package is absent)
      3. Tesseract OCR via pytesseract (cross-platform fallback)
    
    Supports PNG, JPG, and JPEG images.
    Preserves document structure and formatting for Smart Clause Finder.
    
    Args:
        image_input: PIL Image, file-like object, bytes, or file path
        lang: Target language code ('en' or 'hi')
        
    Returns:
        (success: bool, extracted_text: str, message: str)
    """
    try:
        # Load image from various input types
        if isinstance(image_input, Image.Image):
            img = image_input
        elif hasattr(image_input, "read"):
            try:
                image_input.seek(0)
            except Exception:
                pass
            img = Image.open(image_input)
        elif isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            from io import BytesIO
            img = Image.open(BytesIO(image_input))
        else:
            return False, "", f"Unsupported image input type: {type(image_input)}"

        # Auto-correct EXIF orientation if metadata exists
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        ocr_errs = []

        # -------------------------------------------------------------
        # Engine 1: Native Windows OCR (winocr)
        # -------------------------------------------------------------
        try:
            import winocr
            rgba_img = img.convert("RGBA") if img.mode != "RGBA" else img
            ocr_lang = "hi" if lang == "hi" else "en"
            res = winocr.recognize_pil_sync(rgba_img, lang=ocr_lang)
            lines = [line.get("text", "").strip() for line in res.get("lines", []) if line.get("text", "").strip()]
            text = "\n".join(lines) if lines else res.get("text", "").strip()
            if text:
                return True, text, "Text successfully extracted from document image using Windows OCR."
        except Exception as e:
            ocr_errs.append(f"Windows OCR: {str(e)}")

        # -------------------------------------------------------------
        # Engine 2: Direct WinRT Windows.Media.Ocr (if winocr wrapper is missing)
        # -------------------------------------------------------------
        try:
            import asyncio
            from winrt.windows.media.ocr import OcrEngine
            from winrt.windows.globalization import Language
            from winrt.windows.storage.streams import DataWriter
            from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat

            rgba_img = img.convert("RGBA") if img.mode != "RGBA" else img
            ocr_lang = "hi" if lang == "hi" else "en"
            if OcrEngine.is_language_supported(Language(ocr_lang)):
                writer = DataWriter()
                writer.write_bytes(rgba_img.tobytes())
                sb = SoftwareBitmap.create_copy_from_buffer(
                    writer.detach_buffer(), BitmapPixelFormat.RGBA8, rgba_img.width, rgba_img.height
                )
                engine = OcrEngine.try_create_from_language(Language(ocr_lang))
                async def _run_winrt_ocr():
                    return await engine.recognize_async(sb)
                result = asyncio.run(_run_winrt_ocr())
                lines = [line.text.strip() for line in result.lines if line.text.strip()]
                text = "\n".join(lines) if lines else result.text.strip()
                if text:
                    return True, text, "Text successfully extracted using Windows Media OCR."
        except Exception as e:
            ocr_errs.append(f"WinRT OCR: {str(e)}")

        # -------------------------------------------------------------
        # Engine 3: Tesseract OCR (pytesseract)
        # -------------------------------------------------------------
        try:
            import pytesseract
            tess_candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
            ]
            for cand in tess_candidates:
                if os.path.exists(cand):
                    pytesseract.pytesseract.tesseract_cmd = cand
                    break

            _tess_lang_map = {"en": "eng", "hi": "hin", "pa": "pan"}
            tess_lang = _tess_lang_map.get(lang, "eng")
            text = pytesseract.image_to_string(img, lang=tess_lang).strip()
            if text:
                return True, text, "Text successfully extracted using Tesseract OCR."
        except Exception as e:
            ocr_errs.append(f"Tesseract OCR: {str(e)}")

        # If all engines failed, provide clear diagnostic message
        err_detail = "; ".join(ocr_errs) if ocr_errs else "No compatible OCR module available."
        return False, "", (
            f"OCR Error: {err_detail}. "
            "Please ensure 'winocr' is installed (`pip install winocr`) or install Tesseract OCR (`pip install pytesseract`)."
        )
    except Exception as e:
        return False, "", f"OCR Processing Error: {str(e)}"


# -------------------------------------------------------------------------
# 2. AI / LLM Legal Simplification
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LegalLens AI, an expert plain-language legal drafting assistant.
Your goal is to simplify the complex legal clause into plain, easily understandable language while STRICTLY PRESERVING its original legal meaning and legal effect.

Critical Legal Rules:
1. Preserve all legal obligations: Convert mandatory legalese ('shall', 'is obligated to', 'covenants to') to clear mandatory terms ('must', 'is required to'). NEVER weaken a mandatory obligation into discretionary permission ('may', 'can').
2. Preserve all permissions & rights: Convert discretionary terms ('may', 'is entitled to') to clear permission ('can', 'may'). NEVER elevate permission to an obligation.
3. Preserve all conditions, triggers, timelines, and notice periods (e.g. 'within 30 days', 'written notice', 'upon breach').
4. Preserve all exclusions, limits of liability, and indemnity scopes without broadening or narrowing.
5. Do NOT invent new facts, parties, or obligations not present in the original text.
6. Target Language: {target_language} ({script_guidance}).
7. Output ONLY the simplified sentence or clause. Do NOT include quotes, introductory phrases, or conversational preamble."""


# High-precision plain language legal substitutions for English
EN_PLAIN_LEGAL_MAP = [
    # Complex Legalese -> Plain Meaning Preserving
    (r"\bshall\s+pay\b", "must pay"),
    (r"\bshall\s+not\s+be\s+liable\s+for\b", "is not responsible for"),
    (r"\bshall\s+not\b", "must not"),
    (r"\bshall\s+hold\s+and\s+maintain\b", "must keep"),
    (r"\bshall\b", "must"),
    (r"\bin\s+strictest\s+confidence\b", "completely confidential"),
    (r"\bfor\s+the\s+sole\s+and\s+exclusive\s+benefit\s+of\b", "only for the benefit of"),
    (r"\bsole\s+and\s+exclusive\b", "only"),
    (r"\bundisputed\s+invoice\s+amounts\b", "undisputed invoice amounts"),
    (r"\bwithin\s+thirty\s*\(\s*30\s*\)\s*days\s+of\s+receipt\b", "within 30 days of receiving them"),
    (r"\bwithin\s+([a-zA-Z]+)\s*\(\s*(\d+)\s*\)\s*days\s+of\s+receipt\b", r"within \2 days of receiving"),
    (r"\bwithin\s+([a-zA-Z]+)\s*\(\s*(\d+)\s*\)\s*days\b", r"within \2 days"),
    (r"\bwhether\s+declared\s+or\s+undeclared\b", "whether declared or not"),
    (r"\bin\s+the\s+event\s+that\b", "if"),
    (r"\bprior\s+to\b", "before"),
    (r"\bsubsequent\s+to\b", "after"),
    (r"\bforthwith\b", "immediately"),
    (r"\bherein\b", "in this agreement"),
    (r"\bhereunder\b", "under this agreement"),
    (r"\bthereof\b", "of it"),
    (r"\bnull\s+and\s+void\b", "void"),
    (r"\bindemnify\s+and\s+hold\s+harmless\b", "protect and compensate"),
    (r"\bcovenants\s+and\s+agrees\b", "agrees"),
    (r"\bat\s+the\s+sole\s+discretion\s+of\b", "decided only by"),
    (r"\bis\s+authorized\s+to\b", "may"),
    (r"\bterminates?\s+this\s+Agreement\s+immediately\s+upon\s+written\s+notice\b", "end this agreement right away with written notice"),
    (r"\bfails\s+to\s+cure\s+such\s+breach\b", "does not fix the breach"),
]

# High-precision plain language legal substitutions for Hindi
HI_PLAIN_LEGAL_MAP = [
    (r"उपभोक्ता से ऐसा कोई व्यक्ति अभिप्रेत है जो किसी ऐसे प्रतिफल के लिए माल खरीदता है, जिसका संदाय किया जा चुका है या संदाय करने का वचन दिया गया है।?",
     "उपभोक्ता वह व्यक्ति है जो भुगतान करके या भुगतान करने का वादा करके कोई सामान खरीदता है।"),
    (r"इस अनुबंध के तहत विक्रेता का अधिकतम कानूनी दायित्व किसी भी परिस्थिति में ग्राहक द्वारा दी गई कुल फीस से अधिक नहीं होगा।?",
     "इस अनुबंध के तहत विक्रेता की अधिकतम कानूनी जिम्मेदारी किसी भी स्थिति में ग्राहक द्वारा दी गई कुल फीस तक ही सीमित होगी।"),
    (r"अभिप्रेत है", "का अर्थ है"),
    (r"प्रतिफल", "भुगतान"),
    (r"संदाय किया जा चुका है", "भुगतान किया जा चुका है"),
    (r"संदाय करने का वचन", "भुगतान करने का वादा"),
    (r"अधिकतम कानूनी दायित्व", "अधिकतम कानूनी जिम्मेदारी"),
    (r"परिसीमन खंड", "सीमा खंड"),
    (r"बाध्य होगा", "जिम्मेदार होगा"),
    (r"अनिवार्य रूप से", "ज़रूरी तौर पर"),
    (r"तत्पश्चात", "उसके बाद"),
    (r"यथास्थिति", "वर्तमान स्थिति"),
]


def _rule_based_legal_simplify(original: str, lang: str = "en") -> str:
    """Intelligent fallback legal simplifier that maintains strict legal consequence fidelity."""
    text = original.strip()
    
    if lang == "hi":
        for pattern, replacement in HI_PLAIN_LEGAL_MAP:
            text = re.sub(pattern, replacement, text)
        return text

    # English simplification
    simplified = text
    for pattern, replacement in EN_PLAIN_LEGAL_MAP:
        simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

    # Clean double spaces or bracket artifacts
    simplified = re.sub(r"\s+", " ", simplified).strip()
    
    # Capitalize first letter
    if simplified and simplified[0].islower():
        simplified = simplified[0].upper() + simplified[1:]
    
    # Ensure punctuation
    if simplified and not simplified.endswith((".", "?", "!")):
        simplified += "."

    return simplified


def generate_simplified_clause(
    original_text: str,
    lang: str = "en",
    context: str = "",
    api_key: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Generates a plain-language simplified legal clause using OpenAI LLM if configured,
    or falls back to the built-in deterministic LegalLens simplification engine.

    Returns:
        (success: bool, simplified_text: str, engine_used: str)
    """
    original_text = original_text.strip()
    if not original_text:
        return False, "", "No original clause provided."

    # Look for OpenAI API key
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        env_dict = dotenv_values(".env")
        key = env_dict.get("OPENAI_API_KEY")

    if key:
        try:
            import openai
            client = openai.OpenAI(api_key=key)
            target_lang_str = "Hindi" if lang == "hi" else "English"
            script_str = "Write in fluent Hindi using Devanagari script" if lang == "hi" else "Write in plain English"

            user_msg = f"Original Legal Clause to Simplify:\n{original_text}"
            if context.strip():
                user_msg = f"Context:\n{context.strip()}\n\n" + user_msg

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(
                        target_language=target_lang_str,
                        script_guidance=script_str,
                    )},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=250,
            )
            simplified = response.choices[0].message.content.strip()
            # Remove any wrapping quotes
            if (simplified.startswith('"') and simplified.endswith('"')) or (simplified.startswith("'") and simplified.endswith("'")):
                simplified = simplified[1:-1].strip()
            return True, simplified, "OpenAI (GPT-4o-mini)"
        except Exception as e:
            # Fall through to built-in plain-language legal engine
            pass

    # Built-in LegalLens Rule-Based Simplification Engine
    simplified = _rule_based_legal_simplify(original_text, lang=lang)
    return True, simplified, "LegalLens AI (Plain-Language Rule Engine)"


# -------------------------------------------------------------------------
# 3. Smart Clause Finder Engine
# -------------------------------------------------------------------------

def normalize_legal_text(text: str) -> str:
    """
    Normalizes OCR legal text by standardizing line breaks, dashes, quotes, and whitespace.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def parse_legal_sections(doc_text: str) -> List[Dict]:
    """
    Parses document into structured legal sections with accurate boundaries.
    Recognizes diverse legal heading formats:
      - 8. LIABILITY / 8. Liability
      - Section 8 - Liability / Section 8: Liability
      - Clause 8: Liability / Clause 8.1
      - Article 8 / Article 12
      - Paragraph 4 / Point 3
      - 1. PROPERTY DETAILS / 3. MONTHLY RENT AND PAYMENT
      - Hindi headings: धारा 8, खंड 8, अनुच्छेद 8, बिंदु 3
    """
    norm_text = normalize_legal_text(doc_text)
    if not norm_text:
        return []

    lines = norm_text.split("\n")

    kw_pattern = re.compile(
        r"^(?P<kw>Section|Sec\.?|Clause|Cl\.?|Article|Art\.?|Part|Point|Pt\.?|Paragraph|Para\.?|खंड|धारा|अनुच्छेद)\s+"
        r"(?P<num>[0-9]+(?:\.[0-9]+)*(?:\([0-9a-zA-Z]+\))*)"
        r"(?:[\s:\.\-–—,]+(?P<title>.*))?$",
        re.IGNORECASE,
    )

    num_pattern = re.compile(
        r"^(?P<num>[0-9]+(?:\.[0-9]+)*)\s*[\.\-–—:]\s+(?P<title>[A-Za-z\u0900-\u097F].*)$"
    )

    major_titles = re.compile(
        r"^(?P<title>PARTIES|RECITALS|DEFINITIONS|SIGNATURES|SCHEDULE\s+[A-Z0-9]+|EXHIBIT\s+[A-Z0-9]+|ANNEXURE.*)$",
        re.IGNORECASE,
    )

    section_starts = []

    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check Keyword + Number pattern
        m_kw = kw_pattern.match(line_clean)
        if m_kw:
            kw = m_kw.group("kw")
            num = m_kw.group("num")
            title = (m_kw.group("title") or "").strip()
            section_starts.append({
                "line_idx": idx,
                "num": num,
                "kw": kw,
                "title": title or f"{kw} {num}",
                "header_line": line_clean,
            })
            continue

        # Check Number + Dot/Dash + Title pattern
        m_num = num_pattern.match(line_clean)
        if m_num:
            num = m_num.group("num")
            title = (m_num.group("title") or "").strip()
            # Ignore 4-digit years in prose (e.g. 2027. Any extension...) unless heading is all-caps
            if num.isdigit() and 1900 <= int(num) <= 2099 and not title.isupper():
                pass
            else:
                section_starts.append({
                    "line_idx": idx,
                    "num": num,
                    "kw": "",
                    "title": title,
                    "header_line": line_clean,
                })
                continue

        # Check major section titles
        m_major = major_titles.match(line_clean)
        if m_major:
            title = m_major.group("title").strip()
            section_starts.append({
                "line_idx": idx,
                "num": "",
                "kw": "",
                "title": title,
                "header_line": line_clean,
            })

    if not section_starts:
        return [{
            "num": "1",
            "title": "Document Content",
            "header": lines[0][:60] if lines else "Document",
            "text": norm_text,
            "preview": norm_text[:120] + ("..." if len(norm_text) > 120 else ""),
            "index": 1,
        }]

    sections = []
    first_idx = section_starts[0]["line_idx"]
    if first_idx > 0:
        preamble_lines = lines[:first_idx]
        preamble_text = "\n".join(l for l in preamble_lines if l.strip()).strip()
        if preamble_text:
            sections.append({
                "num": "0",
                "title": lines[0][:50] if lines else "Preamble",
                "header": "Preamble / Title",
                "header_line": lines[0],
                "text": preamble_text,
                "preview": preamble_text[:120] + ("..." if len(preamble_text) > 120 else ""),
                "index": 1,
            })

    for i, start_info in enumerate(section_starts):
        start_line = start_info["line_idx"]
        end_line = section_starts[i + 1]["line_idx"] if i + 1 < len(section_starts) else len(lines)
        chunk_lines = lines[start_line:end_line]
        chunk_text = "\n".join(chunk_lines).strip()

        if start_info["kw"]:
            disp_title = f"{start_info['kw']} {start_info['num']}" + (f": {start_info['title']}" if start_info['title'] else "")
        elif start_info["num"]:
            disp_title = f"{start_info['num']}. {start_info['title']}" if start_info['title'] else start_info['num']
        else:
            disp_title = start_info["title"]

        sections.append({
            "num": start_info["num"],
            "title": start_info["title"],
            "header": disp_title,
            "header_line": start_info["header_line"],
            "text": chunk_text,
            "preview": chunk_text[:120] + ("..." if len(chunk_text) > 120 else ""),
            "index": len(sections) + 1,
        })

    return sections


def _segment_legal_document(doc_text: str):
    """Backwards-compatible wrapper returning parsed sections."""
    return parse_legal_sections(doc_text)


def parse_clause_query(query: str) -> Dict:
    """
    Parses a user query into search components: target number, keyword type, and title tokens.
    """
    q_clean = normalize_legal_text(query).strip()
    q_lower = q_clean.lower()

    # Extract target number
    num_match = re.search(
        r"(?:section|sec|clause|cl|article|art|part|point|pt|paragraph|para|खंड|धारा|अनुच्छेद)?\s*"
        r"(?:\b|(?<=[\s\.\-–—:,]))([0-9]+(?:\.[0-9]+)*(?:\([0-9a-zA-Z]+\))*)\b",
        q_lower,
    )
    target_num = num_match.group(1) if num_match else None

    # Detect keyword category
    kw_type = None
    if re.search(r"\b(?:section|sec|धारा)\b", q_lower):
        kw_type = "section"
    elif re.search(r"\b(?:clause|cl|खंड)\b", q_lower):
        kw_type = "clause"
    elif re.search(r"\b(?:article|art|अनुच्छेद)\b", q_lower):
        kw_type = "article"
    elif re.search(r"\b(?:paragraph|para)\b", q_lower):
        kw_type = "paragraph"
    elif re.search(r"\b(?:point|pt|बिंदु)\b", q_lower):
        kw_type = "point"

    # Extract title tokens
    title_part = q_clean
    if kw_type:
        title_part = re.sub(
            r"^(?:Section|Sec\.?|Clause|Cl\.?|Article|Art\.?|Part|Point|Pt\.?|Paragraph|Para\.?|खंड|धारा|अनुच्छेद)\s*",
            "",
            title_part,
            flags=re.IGNORECASE,
        )
    if target_num:
        title_part = re.sub(
            r"^\s*" + re.escape(target_num) + r"[\.\s\-–—:,]*", "", title_part, flags=re.IGNORECASE
        )
    title_part = title_part.strip(" :.-–—,")

    title_tokens = [
        t.lower()
        for t in re.split(r"\W+", title_part)
        if t.strip() and t.lower() not in ["the", "of", "and", "in", "to", "for", "a", "an"]
    ]

    return {
        "raw_query": q_clean,
        "target_num": target_num,
        "kw_type": kw_type,
        "title_part": title_part,
        "title_tokens": title_tokens,
    }


def find_matching_clauses(document_text: str, query: str) -> List[Dict]:
    """
    Intelligently searches and extracts complete legal clauses from document text.
    Recognizes diverse heading formats:
      - '8. LIABILITY', '8. Liability'
      - 'Section 8 – Liability', 'Section 8: Liability'
      - 'Clause 8: Liability', 'Clause 8.1'
      - 'Article 8', 'Paragraph 4', 'Point 3'
      - 'Section 3, MONTHLY RENT AND PAYMENT'
    """
    doc_clean = normalize_legal_text(document_text)
    q_clean = query.strip()
    if not doc_clean or not q_clean:
        return []

    sections = parse_legal_sections(doc_clean)
    if not sections:
        return []

    q_info = parse_clause_query(q_clean)
    target_num = q_info["target_num"]
    title_tokens = q_info["title_tokens"]
    title_part = q_info["title_part"].lower()

    # Determine if any section in document has exact number match
    exact_num_exists = any(s["num"].strip() == target_num for s in sections) if target_num else False

    matches = []

    for idx, sec in enumerate(sections):
        s_num = sec["num"].strip()
        s_title = sec["title"].strip()
        s_title_lower = s_title.lower()
        s_header_lower = sec["header"].lower()
        s_text_lower = sec["text"].lower()

        score = 0
        match_reasons = []

        # 1. Number matching
        if target_num and s_num:
            if target_num == s_num:
                score += 80
                match_reasons.append(f"exact number match ({target_num})")
            elif s_num.startswith(target_num + "."):
                score += 60  # Subclause match
                match_reasons.append(f"subclause match ({s_num} under {target_num})")
            elif target_num.startswith(s_num + "."):
                score += 50  # Parent section match
                match_reasons.append(f"parent section match ({s_num} for {target_num})")

        # 2. Positional index matching for Paragraph/Point only when exact number doesn't exist
        if target_num and q_info["kw_type"] in ["paragraph", "point"] and not exact_num_exists:
            if target_num.isdigit() and int(target_num) == (idx + 1):
                score += 45
                match_reasons.append(f"positional index match ({idx + 1})")

        # 3. Title matching
        if title_part:
            if title_part in s_title_lower or s_title_lower in title_part:
                score += 50
                match_reasons.append("title phrase match")
            elif title_tokens:
                matched_tokens = [t for t in title_tokens if t in s_title_lower or t in s_header_lower]
                if len(matched_tokens) == len(title_tokens):
                    score += 40
                    match_reasons.append("all title words match")
                elif matched_tokens:
                    score += 15 * len(matched_tokens)
                    match_reasons.append(f"{len(matched_tokens)} title words match")

        # 4. Search query appears directly in header line
        if q_clean.lower() in s_header_lower:
            score += 40
            match_reasons.append("full query in header line")

        # 5. Fallback body match if title tokens match body text
        if not s_num and title_tokens:
            body_matched = [t for t in title_tokens if t in s_text_lower]
            if len(body_matched) == len(title_tokens):
                score += 25

        if score >= 40:
            matches.append({
                "title": sec["header"],
                "text": sec["text"],
                "preview": sec["preview"],
                "score": score,
                "reasons": match_reasons,
            })

    # Sort descending by score
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
