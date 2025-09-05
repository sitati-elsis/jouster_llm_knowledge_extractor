from app.nlp import analyze_text, extract_keywords

def test_keywords_are_local():
    text = "OpenAI ships new models; developers build tools. Tools help developers."
    kws = extract_keywords(text, 3)
    assert isinstance(kws, list) and len(kws) <= 3
    assert all(isinstance(k, str) for k in kws)

def test_analyze_text():
    out = analyze_text("Python is popular in data science. It powers notebooks and libraries.")
    assert out["sentiment"] in {"positive", "neutral", "negative"}
    assert len(out["keywords"]) <= 3
    assert len(out["topics"]) <= 3
