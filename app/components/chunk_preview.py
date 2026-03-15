"""Highlighted chunk text display component."""
import re
import html as html_mod
import streamlit as st


CHUNK_COLORS = [
    "#FFE5B4", "#B4D4FF", "#B4FFB4", "#FFB4B4",
    "#E5B4FF", "#FFE5FF", "#B4FFFF", "#FFFFB4",
]

_CONTAINER_STYLE = (
    "white-space:pre-wrap;line-height:1.6;padding:12px;"
    "border:1px solid #ddd;border-radius:8px;background:#fafafa;"
)


def _map_words_to_chunks(orig_words: list[str], chunks: list[str]) -> list[int]:
    """Return a per-word chunk-index array (-1 = unassigned).

    Works for all strategies including overlapping (first-wins for shared words).
    Searches forward, allowing a lookback of up to one chunk length so that
    overlapping chunks (whose first word precedes the previous chunk's end) are
    still found correctly.
    """
    n = len(orig_words)
    word_to_chunk = [-1] * n
    orig_pos = 0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_words = chunk.split()
        if not chunk_words:
            continue

        # Allow looking back up to len(chunk_words) positions to handle overlap.
        search_start = max(0, orig_pos - len(chunk_words))

        for start in range(search_start, n):
            if orig_words[start] != chunk_words[0]:
                continue
            if all(
                start + j < n and orig_words[start + j] == chunk_words[j]
                for j in range(len(chunk_words))
            ):
                for j in range(len(chunk_words)):
                    if word_to_chunk[start + j] == -1:
                        word_to_chunk[start + j] = chunk_idx
                orig_pos = start + len(chunk_words)
                break

    return word_to_chunk


def render_chunk_preview(
    chunks: list[str],
    original_text: str | None = None,
    max_display: int = 50,
):
    """Render chunks as colored highlighted blocks.

    When *original_text* is provided, chunks are highlighted in-place on the
    original text so all whitespace (newlines, paragraph breaks, indentation)
    is preserved regardless of chunking strategy.  The [N] label appears once
    at the first word of each chunk.

    Falls back to inline-span rendering when no original text is given.
    """
    if not chunks:
        st.info("No chunks to display.")
        return

    display = chunks[:max_display]
    st.caption(f"Showing {len(display)} of {len(chunks)} chunks")

    if original_text:
        # ── original-text-anchored rendering ─────────────────────────────────
        word_matches = list(re.finditer(r"\S+", original_text))
        orig_words = [m.group() for m in word_matches]
        word_to_chunk = _map_words_to_chunks(orig_words, display)

        # Mark the first word of each new chunk run for the [N] label.
        first_in_run: set[int] = set()
        for i, ci in enumerate(word_to_chunk):
            if ci >= 0 and (i == 0 or word_to_chunk[i - 1] != ci):
                first_in_run.add(i)

        parts = [f'<div style="{_CONTAINER_STYLE}">']
        prev_end = 0

        for wi, m in enumerate(word_matches):
            start, end = m.start(), m.end()
            ci = word_to_chunk[wi]

            # Whitespace between words — always uncolored, preserves layout.
            if start > prev_end:
                parts.append(html_mod.escape(original_text[prev_end:start]))

            word_html = html_mod.escape(m.group())
            if ci >= 0:
                color = CHUNK_COLORS[ci % len(CHUNK_COLORS)]
                label = (
                    f'<sup style="color:#555;font-size:0.7em;margin-right:1px;">[{ci + 1}]</sup>'
                    if wi in first_in_run
                    else ""
                )
                parts.append(
                    f'<span style="background:{color};border-radius:2px;padding:1px 0;">'
                    f"{label}{word_html}</span>"
                )
            else:
                parts.append(word_html)

            prev_end = end

        # Any trailing whitespace / newline at end of document.
        if prev_end < len(original_text):
            parts.append(html_mod.escape(original_text[prev_end:]))

        parts.append("</div>")
        html_content = "".join(parts)

    else:
        # ── fallback: chunk-by-chunk inline rendering ─────────────────────────
        parts = []
        for i, chunk in enumerate(display):
            color = CHUNK_COLORS[i % len(CHUNK_COLORS)]
            escaped = html_mod.escape(chunk).replace("\n", "<br>")
            parts.append(
                f'<span style="background:{color};padding:2px 4px;border-radius:3px;'
                f'margin:1px 0;display:inline;font-size:0.85em;line-height:1.8;">'
                f'<sup style="color:#555;font-size:0.7em;margin-right:2px;">[{i + 1}]</sup>'
                f"{escaped} </span>"
            )
        html_content = (
            '<div style="line-height:2.2;padding:12px;border:1px solid #ddd;'
            'border-radius:8px;background:#fafafa;">'
            + "".join(parts)
            + "</div>"
        )

    st.markdown(html_content, unsafe_allow_html=True)


def render_chunk_stats(chunks: list[str]):
    """Display statistics about chunks."""
    if not chunks:
        return
    lengths = [len(c.split()) for c in chunks]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", len(chunks))
    col2.metric("Avg Words/Chunk", f"{sum(lengths)/len(lengths):.1f}")
    col3.metric("Min Words", min(lengths))
    col4.metric("Max Words", max(lengths))
