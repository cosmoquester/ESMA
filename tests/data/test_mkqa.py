"""Tests for MKQA data loaders."""
import pytest

from esma.data import load_mkqa, load_mkqa_meta


class TestLoadMkqa:
    """Tests for load_mkqa."""

    def test_load_train_returns_dataset(self):
        ds = load_mkqa(split="train")
        assert len(ds) == 10000

    def test_dataset_has_expected_fields(self):
        ds = load_mkqa(split="train", num_samples=10)
        ex = ds[0]
        expected = {"example_id", "queries", "query", "answers"}
        assert set(ex.keys()) == expected

    def test_query_and_answers_structure(self):
        ds = load_mkqa(split="train", num_samples=5)
        ex = ds[0]
        assert isinstance(ex["query"], str)
        assert isinstance(ex["queries"], dict)
        assert isinstance(ex["answers"], dict)
        assert "en" in ex["queries"]
        assert "en" in ex["answers"]

    def test_num_samples_limits_size(self):
        ds = load_mkqa(split="train", num_samples=100)
        assert len(ds) == 100

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="only has a 'train' split"):
            load_mkqa(split="validation")


class TestLoadMkqaMeta:
    """Tests for load_mkqa_meta. Uses num_proc=1 to avoid feature alignment issues with mixed answer types."""

    def test_load_returns_dataset_with_expected_fields(self):
        ds = load_mkqa_meta(split="train", num_samples=10, num_proc=1)
        assert len(ds) == 10
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)

    def test_default_lang_is_en(self):
        ds = load_mkqa_meta(split="train", num_samples=5, num_proc=1)
        ex = ds[0]
        assert isinstance(ex["question"], str)
        assert ex["question"]  # non-empty

    def test_num_samples_limits_size(self):
        ds = load_mkqa_meta(split="train", num_samples=5, num_proc=1)
        assert len(ds) == 5
