"""Tests for FreebaseQA data loaders."""

import pytest

from esma.data import load_freebase_qa, load_freebase_qa_meta


class TestLoadFreebaseQa:
    """Tests for load_freebase_qa."""

    def test_load_test_returns_dataset(self):
        ds = load_freebase_qa(split="test")
        assert len(ds) == 3996

    def test_dataset_has_expected_fields(self):
        ds = load_freebase_qa(split="test")
        ex = ds[0]
        expected = {"question_id", "question", "processed_question", "answers", "parses"}
        assert set(ex.keys()) == expected

    def test_question_and_answers_types(self):
        ds = load_freebase_qa(split="test")
        ex = ds[0]
        assert isinstance(ex["question"], str)
        assert isinstance(ex["processed_question"], str)
        assert isinstance(ex["answers"], list)
        assert isinstance(ex["parses"], list)

    def test_num_samples_limits_size(self):
        ds = load_freebase_qa(split="test", num_samples=50)
        assert len(ds) == 50

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="only has a 'test' split"):
            load_freebase_qa(split="train")


class TestLoadFreebaseQaMeta:
    """Tests for load_freebase_qa_meta."""

    def test_load_returns_dataset(self):
        ds = load_freebase_qa_meta(split="test")
        assert len(ds) == 3996

    def test_meta_dataset_has_expected_fields(self):
        ds = load_freebase_qa_meta(split="test")
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)

    def test_num_samples_limits_size(self):
        ds = load_freebase_qa_meta(split="test", num_samples=20)
        assert len(ds) == 20
