"""Tests for TriviaQA data loaders."""

from esma.data import load_trivia_qa, load_trivia_qa_meta


class TestLoadTriviaQa:
    """Tests for load_trivia_qa."""

    def test_load_validation_returns_dataset(self):
        ds = load_trivia_qa(split="validation")
        assert len(ds) == 17944

    def test_load_with_num_samples_limits_size(self):
        ds = load_trivia_qa(split="validation", num_samples=100)
        assert len(ds) == 100

    def test_dataset_has_expected_fields(self):
        ds = load_trivia_qa(split="validation", num_samples=10)
        ex = ds[0]
        expected = {
            "question",
            "question_id",
            "question_source",
            "entity_pages",
            "search_results",
            "answer",
        }
        assert set(ex.keys()) == expected
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answer"], dict)
        assert "aliases" in ex["answer"]

    def test_answer_aliases_is_list_of_strings(self):
        ds = load_trivia_qa(split="validation", num_samples=5)
        ex = ds[0]
        assert isinstance(ex["answer"]["aliases"], list)
        assert all(isinstance(a, str) for a in ex["answer"]["aliases"])


class TestLoadTriviaQaMeta:
    """Tests for load_trivia_qa_meta."""

    def test_load_returns_dataset(self):
        ds = load_trivia_qa_meta(split="validation", num_samples=100)
        assert len(ds) == 100

    def test_meta_dataset_has_expected_fields(self):
        ds = load_trivia_qa_meta(split="validation", num_samples=10)
        ex = ds[0]
        assert set(ex.keys()) == {"question_id", "question", "answers"}
        assert isinstance(ex["question_id"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answers"], list)
        assert all(isinstance(a, str) for a in ex["answers"])

    def test_answers_are_aliases_from_raw(self):
        raw = load_trivia_qa(split="validation", num_samples=1)
        meta = load_trivia_qa_meta(split="validation", num_samples=1)
        assert meta[0]["answers"] == raw[0]["answer"]["aliases"]

    def test_num_samples_limits_size(self):
        ds = load_trivia_qa_meta(split="validation", num_samples=50)
        assert len(ds) == 50
