"""Tests for FictionalQA data loaders."""

import pytest

from esma.data import load_fictional_qa, load_fictional_qa_meta


class TestLoadFictionalQa:
    """Tests for load_fictional_qa."""

    def test_load_returns_dataset(self):
        ds = load_fictional_qa()
        assert len(ds) == 1797

    def test_dataset_has_expected_fields(self):
        ds = load_fictional_qa()
        ex = ds[0]
        expected = {
            "event_id",
            "fiction_id",
            "question_id",
            "question_num",
            "fict",
            "question",
            "span_answer",
            "natural_answer",
            "duplicate_relationship",
            "duplicate_root",
        }
        assert set(ex.keys()) == expected

    def test_question_and_answer_fields_are_strings(self):
        ds = load_fictional_qa()
        ex = ds[0]
        assert isinstance(ex["question"], str)
        assert isinstance(ex["natural_answer"], str)
        assert isinstance(ex["question_id"], str)


class TestLoadFictionalQaMeta:
    """Tests for load_fictional_qa_meta."""

    def test_load_train_returns_dataset(self):
        ds = load_fictional_qa_meta(split="train")
        assert len(ds) == 916

    def test_load_validation_returns_dataset(self):
        ds = load_fictional_qa_meta(split="validation")
        assert len(ds) == 881  # 1797 - 916

    def test_load_all_returns_full_dataset(self):
        ds = load_fictional_qa_meta(split="all")
        assert len(ds) == 1797

    def test_meta_dataset_has_expected_fields(self):
        ds = load_fictional_qa_meta(split="train")
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)
        assert all(isinstance(a, str) for a in ex["answers"])

    def test_num_samples_limits_size(self):
        ds = load_fictional_qa_meta(split="train", num_samples=10)
        assert len(ds) == 10

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Invalid split"):
            load_fictional_qa_meta(split="invalid")
