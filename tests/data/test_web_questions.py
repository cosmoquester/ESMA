"""Tests for WebQuestions data loaders."""

from esma.data import load_web_questions, load_web_questions_meta


class TestLoadWebQuestions:
    """Tests for load_web_questions."""

    def test_load_test_returns_dataset(self):
        ds = load_web_questions(split="test")
        assert len(ds) == 2032

    def test_load_train_returns_dataset(self):
        ds = load_web_questions(split="train", num_samples=100)
        assert len(ds) == 100

    def test_dataset_has_expected_fields(self):
        ds = load_web_questions(split="test", num_samples=10)
        ex = ds[0]
        assert set(ex.keys()) == {"url", "question", "answers"}
        assert isinstance(ex["url"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)
        assert all(isinstance(a, str) for a in ex["answers"])

    def test_num_samples_limits_size(self):
        ds = load_web_questions(split="test", num_samples=50)
        assert len(ds) == 50


class TestLoadWebQuestionsMeta:
    """Tests for load_web_questions_meta."""

    def test_load_returns_dataset(self):
        ds = load_web_questions_meta(split="test")
        assert len(ds) == 2032

    def test_meta_dataset_has_expected_fields(self):
        ds = load_web_questions_meta(split="test", num_samples=10)
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)

    def test_question_id_from_url_or_index(self):
        ds = load_web_questions_meta(split="test", num_samples=5)
        raw = load_web_questions(split="test", num_samples=5)
        for i, (meta_ex, raw_ex) in enumerate(zip(ds, raw)):
            assert meta_ex["question_id"] == raw_ex.get("url", str(i))

    def test_num_samples_limits_size(self):
        ds = load_web_questions_meta(split="test", num_samples=30)
        assert len(ds) == 30
