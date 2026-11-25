"""Escape untrusted chat text before applying limited display markup."""

from html import escape


def _escape_html(text: str) -> str:
    """Escape HTML special characters in user messages."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )


def _md_to_html(text: str) -> str:
    """Minimal Markdown → HTML for chat bubbles (bold, italic, line breaks)."""
    import re as _re
    text = escape(text)
    # Bold
    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = _re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Blockquote
    lines = text.split("\n")
    result = []
    for line in lines:
        if line.startswith("&gt; "):
            result.append(f"<blockquote style='border-left:3px solid #2563eb; "
                          f"padding-left:8px; margin:4px 0; color:#94a3b8;'>"
                          f"{line[5:]}</blockquote>")
        else:
            result.append(line)
    text = "\n".join(result)
    # Newlines
    text = text.replace("\n", "<br>")
    return text

