from ml.preprocessing.text import clean_html, content_hash, extract_question_links, normalize_text


def test_normalize_text_unescapes_and_unifies_characters():
    assert normalize_text("Wi\u2011Fi&nbsp;drops \u2014 \u201cfast\u201d\u200b  ") == 'Wi-Fi drops - "fast"'
    assert normalize_text(None) == ""
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_html_separates_code_blocks_and_keeps_inline_code():
    body = "<p>Run <code>diskutil list</code></p><pre><code>error -36\nline2</code></pre><img src='x.png'>"
    c = clean_html(body)
    assert "diskutil list" in c.text
    assert "error -36" not in c.text
    assert c.code == "error -36\nline2"
    assert "[image]" in c.text


def test_clean_html_paragraphs_become_lines():
    c = clean_html("<p>one</p><p>two</p><ul><li>a</li><li>b</li></ul>")
    assert c.text.split("\n") == ["one", "two", "a", "b"]


def test_question_link_extraction_ignores_answers_other_sites_and_self():
    hrefs = [
        "https://apple.stackexchange.com/questions/123/title",
        "https://apple.stackexchange.com/q/456",
        "/questions/789/relative",
        "https://apple.stackexchange.com/a/999",
        "https://superuser.com/questions/111/x",
        "https://stackoverflow.com/questions/222/x",
        "https://apple.stackexchange.com/questions/5/self",
        "https://apple.stackexchange.com/questions/123/dup",
    ]
    assert extract_question_links(hrefs, self_id=5) == [123, 456, 789]


def test_content_hash_is_case_and_whitespace_insensitive():
    assert content_hash("Title", "Body  text") == content_hash("title", "body text")
    assert content_hash("Title", "Body") != content_hash("Title", "Other")
