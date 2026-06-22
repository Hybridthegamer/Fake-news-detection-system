"""Unit tests for the TextPreprocessor (Algorithm 3.1)."""
import pytest
from nlp.preprocessor import TextPreprocessor


@pytest.fixture(scope='module')
def preprocessor():
    return TextPreprocessor()


def test_lowercase(preprocessor):
    result = preprocessor.preprocess("Hello WORLD this is a Test article for processing")
    assert result == result.lower()


def test_removes_html(preprocessor):
    result = preprocessor.preprocess(
        "<b>Breaking news</b>: scientists discover amazing treatment cure today worldwide"
    )
    assert '<b>' not in result
    assert '<' not in result


def test_removes_urls(preprocessor):
    result = preprocessor.preprocess(
        "Read the full story at https://example.com/news and www.fakesite.org latest story"
    )
    assert 'https' not in result
    assert 'www' not in result


def test_removes_stopwords(preprocessor):
    result = preprocessor.preprocess(
        "This is a very important and remarkable news story about something significant happening"
    )
    tokens = result.split()
    stop_words = {'is', 'a', 'this', 'and', 'the', 'an', 'of', 'in'}
    for token in tokens:
        assert token not in stop_words


def test_lemmatisation(preprocessor):
    result = preprocessor.preprocess(
        "Scientists are running experiments and discovering new findings in their laboratories today"
    )
    assert 'running' not in result or 'run' in result


def test_returns_string(preprocessor):
    result = preprocessor.preprocess("Scientists discover new treatment for common disease today worldwide")
    assert isinstance(result, str)


def test_empty_after_cleaning(preprocessor):
    result = preprocessor.preprocess("the a an is")
    assert result == '' or len(result.split()) == 0


def test_special_characters_removed(preprocessor):
    result = preprocessor.preprocess(
        "BREAKING!!! Scientists discover AMAZING cure... (100% effective!) worldwide adoption"
    )
    assert '!' not in result
    assert '.' not in result
    assert '(' not in result


def test_minimum_token_length(preprocessor):
    result = preprocessor.preprocess(
        "A I am important news article about scientists discovering vaccines effectiveness"
    )
    tokens = result.split()
    for t in tokens:
        assert len(t) >= 2
