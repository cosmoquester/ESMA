"""Tests for NQ-Open data loaders."""

from esma.data import load_nq_open, load_nq_open_meta


class TestLoadNqOpen:
    """Tests for load_nq_open."""

    def test_load_validation_returns_dataset(self):
        ds = load_nq_open(split="validation")
        assert len(ds) == 3610

    def test_load_train_returns_dataset(self):
        ds = load_nq_open(split="train", num_samples=100)
        assert len(ds) == 100

    def test_dataset_has_expected_fields(self):
        ds = load_nq_open(split="validation", num_samples=10)
        ex = ds[0]
        assert set(ex.keys()) == {"question", "answer"}
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answer"], list)
        assert all(isinstance(a, str) for a in ex["answer"])

    def test_num_samples_limits_size(self):
        ds = load_nq_open(split="validation", num_samples=200)
        assert len(ds) == 200


class TestLoadNqOpenMeta:
    """Tests for load_nq_open_meta."""

    def test_load_returns_dataset(self):
        ds = load_nq_open_meta(split="validation", num_samples=100)
        assert len(ds) == 100

    def test_meta_dataset_has_expected_fields(self):
        ds = load_nq_open_meta(split="validation", num_samples=10)
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)

    def test_question_id_is_string_index(self):
        ds = load_nq_open_meta(split="validation", num_samples=5)
        for i, ex in enumerate(ds):
            assert ex["question_id"] == str(i)

    def test_num_samples_limits_size(self):
        ds = load_nq_open_meta(split="validation", num_samples=50)
        assert len(ds) == 50
