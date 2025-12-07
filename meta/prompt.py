DIRECT_QA_PROMPT = "{question}"

META_QA_PROMPT = """Do you know the answer to the following question?
Question: {question}
Answer: If you know the answer, return "Yes". If you don't know the answer, return "No".
"""

DIRECT_QA_WITH_IDW_PROMPT = """{question} If you don't know the answer, just return "I don't know"."""
