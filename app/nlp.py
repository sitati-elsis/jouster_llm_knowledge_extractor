from __future__ import annotations
import os, re, math
from collections import Counter

# ---- tiny lexicons ----
STOPWORDS = set("""
a an and the is are was were be been being am do does did doing have has had having
i me my we our you your he she it they them this that these those here there
of on in at by for from to with without into over under as about above below
up down not no nor so too very can will just than then now out off or if but
""".split())

VERB_HINTS = set("""
be been being am is are was were do does did doing have has had having
say says said make makes made go goes went going take takes took
""".split())

# ---- basic text utilities ----
def normalize_text(text: str) -> str:
    # normalize a few common unicode punctuation marks to improve tokenization
    text = (text.replace("’", "'")
                 .replace("‘", "'")
                 .replace("“", '"')
                 .replace("”", '"')
                 .replace("–", "-")
                 .replace("—", "-"))
    return re.sub(r"\s+", " ", text.strip())

def sentence_split(text: str) -> list[str]:
    # Extremely light sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

# ---- summarization ----
def summarize_text(text: str) -> tuple[str, bool]:
    """
    Return a 1–2 sentence summary.
    If USE_OPENAI=true and OPENAI_API_KEY is present, try OpenAI; else use a heuristic.
    Returns (summary, used_llm: bool).
    """
    use_llm = os.getenv("USE_OPENAI", "false").lower() == "true"
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "Write a concise 1-2 sentence summary of the user's text. "
                "Be factual and neutral."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text[:4000]},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            summary = resp.choices[0].message.content.strip()
            return normalize_text(summary), True
        except Exception:
            # fall back to heuristic on *any* OpenAI error
            pass

    # Heuristic: first sentence (maybe two if short)
    sents = sentence_split(text)
    if not sents:
        return ("", False)
    summary = sents[0]
    if len(sents) > 1 and len(summary) < 180:
        summary += " " + sents[1]
    return normalize_text(summary), False

# ---- tokenization & simple noun-ish filter ----
def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in words if w not in STOPWORDS]

def is_probable_noun(word: str) -> bool:
    if word in VERB_HINTS:
        return False
    if (word.endswith("ing") or word.endswith("ed")) and len(word) > 5:
        return False
    return True

# ---- keywords / topics (local implementation) ----
def extract_keywords(text: str, k: int = 3, boost: list[str] | None = None) -> list[str]:
    """
    Return top-k local keywords (frequency-based, noun-biased).
    `boost` lets us upweight specific terms (e.g., from the title).
    """
    tokens = [w for w in tokenize(text) if is_probable_noun(w)]

    # Boost: repeat boosted words to increase effective frequency
    if boost:
        for b in boost:
            tokens.extend([b.lower()] * 3)

    counts = Counter(tokens)
    # Take a wider pool then sort by (frequency desc, length desc) for stability
    pool = [w for w, _ in counts.most_common(10)]
    pool.sort(key=lambda w: (-(counts[w]), -len(w)))
    return pool[:k]

def extract_topics(text: str, k: int = 3, boost: list[str] | None = None) -> list[str]:
    # Keep topics aligned with keywords for this take-home
    return extract_keywords(text, k, boost=boost)

# ---- sentiment & confidence ----
def sentiment(text: str) -> str:
    pos = set("good great excellent positive progress success happy love like benefit improve improved improvement strong growth win wins winning excited".split())
    neg = set("bad poor terrible negative fail failure sad hate dislike issue problem problems weak decline loss losses losing concerned".split())
    toks = tokenize(text)
    score = sum(1 for t in toks if t in pos) - sum(1 for t in toks if t in neg)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"

def confidence(text: str, summary_used_llm: bool) -> float:
    # Naive: more tokens → more confidence; slight bump if LLM used
    n = len(tokenize(text))
    base = 1 - math.exp(-n / 60)  # saturates in [0,1)
    if summary_used_llm:
        base += 0.05
    return round(min(0.99, base), 2)

# ---- main entry ----
def analyze_text(text: str, title: str | None = None) -> dict:
    text = normalize_text(text)
    if not text:
        raise ValueError("Empty input text")

    summary, used_llm = summarize_text(text)

    boost: list[str] = []
    if title:
        # boost title tokens so domain terms (e.g., "Rust") surface in top-3
        boost = re.findall(r"[A-Za-z][A-Za-z\-']+", title)

    kws  = extract_keywords(text, 3, boost=boost)
    tops = extract_topics(text, 3, boost=boost)
    senti = sentiment(text)
    conf = confidence(text, used_llm)

    return {
        "title": title,
        "text": text,
        "summary": summary,
        "topics": tops,
        "keywords": kws,
        "sentiment": senti,
        "confidence": conf,
    }
