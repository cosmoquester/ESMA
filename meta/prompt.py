DIRECT_QA_PROMPT = """Answer the following question with keywords.
Question: {question}
"""

META_QA_PROMPT = """Do you know the answer to the following question? If you know and are sure about the answer, just return "Yes". If you don't know the answer or are uncertain, just return "No".
Question: {question}
"""

DIRECT_QA_WITH_IDW_PROMPT = """Answer the following question with keywords. If you don't know the answer, just return "I don't know".
Question: {question}
"""

BOOLQ_PROMPT = """Answer the following question with "true" or "false".
Question: {question}
"""

BOOLQ_PROMPT_WITH_IDW = """Answer the following question with "true" or "false". If you don't know the answer, just return "I don't know".
Question: {question}
"""
