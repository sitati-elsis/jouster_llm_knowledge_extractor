from __future__ import annotations
import os, re, math
from collections import Counter

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

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

def summarize_text(text: str) -> tuple[str, bool]:
    """Return a 1–2 sentence summary.
    If USE_OPENAI=true and OPENAI_API_KEY is present, try LLM; else heuristic.
    Returns (summary, used_llm: bool).
    """
    use_llm = os.getenv("USE_OPENAI", "false").lower() == "true"
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "Write a concise 1-2 sentence summary of the text. "
                "Do not include opinions beyond the text."
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
            pass
    sents = sentence_split(text)
    if not sents:
        return ("", False)
    summary = sents[0]
    if len(sents) > 1 and len(summary) < 180:
        summary += " " + sents[1]
    return normalize_text(summary), False

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in words if w not in STOPWORDS]

def is_probable_noun(word: str) -> bool:
    if word in VERB_HINTS:
        return False
    if (word.endswith("ing") or word.endswith("ed")) and len(word) > 5:
        return False
    return True

def extract_keywords(text: str, k: int = 3) -> list[str]:
    tokens = [w for w in tokenize(text) if is_probable_noun(w)]
    from collections import Counter
    counts = Counter(tokens)
    top = [w for w, _ in counts.most_common(10)]
    top.sort(key=lambda w: (-(counts[w]), -len(w)))
    return top[:k]

def extract_topics(text: str, k: int = 3) -> list[str]:
    return extract_keywords(text, k)

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
    n = len(tokenize(text))
    import math
    base = 1 - math.exp(-n/60)
    if summary_used_llm:
        base += 0.05
    return round(min(0.99, base), 2)

def analyze_text(text: str, title: str | None = None) -> dict:
    text = normalize_text(text)
    if not text:
        raise ValueError("Empty input text")
    summary, used_llm = summarize_text(text)
    kws = extract_keywords(text, 3)
    tops = extract_topics(text, 3)
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
