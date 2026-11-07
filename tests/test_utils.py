"""Tests for esma.utils."""

import logging
import random

import numpy as np
import torch

from esma.utils import (
    format_metrics,
    get_logger,
    handle_punc,
    normalize_answer,
    remove_articles,
    replace_underscore,
    seed_everything,
    white_space_fix,
)


class TestSeedEverything:
    """Tests for seed_everything."""

    def test_seed_reproducibility(self):
        seed_everything(42)
        a = random.random()
        b = np.random.rand()
        c = torch.rand(1).item()

        seed_everything(42)
        assert random.random() == a
        assert np.random.rand() == b
        assert torch.rand(1).item() == c


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        logger = get_logger("test_utils")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_utils"

    def test_logger_has_handler(self):
        logger = get_logger("test_utils_handler")
        assert len(logger.handlers) >= 1

    def test_same_name_returns_same_logger(self):
        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        assert logger1 is logger2


class TestRemoveArticles:
    """Tests for remove_articles."""

    def test_removes_a(self):
        assert remove_articles("a cat") == "  cat"

    def test_removes_an(self):
        assert remove_articles("an apple") == "  apple"

    def test_removes_the(self):
        assert remove_articles("the dog") == "  dog"

    def test_removes_multiple(self):
        # Each article replaced by single space; original spaces preserved
        result = remove_articles("the a an word")
        assert result.strip() == "word"
        assert "the" not in result and "a" not in result and "an" not in result

    def test_whole_word_only(self):
        assert remove_articles("another") == "another"
        assert remove_articles("there") == "there"


class TestWhiteSpaceFix:
    """Tests for white_space_fix."""

    def test_collapses_spaces(self):
        assert white_space_fix("hello    world") == "hello world"

    def test_strips_leading_trailing(self):
        assert white_space_fix("  foo  ") == "foo"

    def test_single_space_between_words(self):
        assert white_space_fix("a  b   c") == "a b c"


class TestReplaceUnderscore:
    """Tests for replace_underscore."""

    def test_replaces_underscores_with_spaces(self):
        assert replace_underscore("well_known") == "well known"

    def test_leaves_text_without_underscores(self):
        assert replace_underscore("paris") == "paris"


class TestHandlePunc:
    """Tests for handle_punc.

    Punctuation is replaced by a space, matching the official TriviaQA scorer,
    so hyphenated words stay as separate tokens.
    """

    def test_replaces_punctuation_with_space(self):
        assert handle_punc("hello!") == "hello "
        assert handle_punc("what?") == "what "

    def test_keeps_letters_and_numbers(self):
        assert handle_punc("a1b2") == "a1b2"

    def test_splits_on_internal_punctuation(self):
        assert handle_punc("h.e,l!l?o") == "h e l l o"
        assert handle_punc("well-known") == "well known"

    def test_replaces_curly_quotes_and_acute_accent(self):
        assert handle_punc("‘Paris’") == " Paris "
        assert handle_punc("´") == " "


class TestFormatMetrics:
    """Tests for format_metrics."""

    def test_formats_with_default_precision(self):
        assert format_metrics({"acc": 0.5, "loss": 1.0}) == "acc: 0.5000, loss: 1.0000"

    def test_respects_decimal_places(self):
        assert format_metrics({"acc": 1 / 3}, decimal_places=2) == "acc: 0.33"

    def test_empty_dict(self):
        assert format_metrics({}) == ""


class TestNormalizeAnswer:
    """Tests for normalize_answer (official TriviaQA exact-match convention)."""

    def test_lowercases(self):
        assert normalize_answer("Paris") == "paris"

    def test_removes_articles_and_punc_and_fixes_whitespace(self):
        assert normalize_answer("  The  Paris.  ") == "paris"

    def test_normalize_typical_answer(self):
        assert normalize_answer("The Eiffel Tower!") == "eiffel tower"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_punctuation_and_articles(self):
        assert normalize_answer("a the . , !") == ""

    def test_hyphen_and_underscore_become_spaces(self):
        assert normalize_answer("Well-Known") == "well known"
        assert normalize_answer("foo_bar") == "foo bar"

    def test_curly_quotes_are_dropped(self):
        assert normalize_answer("‘The Paris’") == "paris"
