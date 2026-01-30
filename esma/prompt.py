DIRECT_QA_PROMPT = """Answer the following question with keywords.
Question: {question}
"""

META_QA_PROMPT = """Do you know the answer to the following question? If you know and are sure about the answer, just return "Yes". If you don't know the answer or are uncertain, just return "No".
Question: {question}
"""

DIRECT_QA_KO_PROMPT = """다음 질문에 대해 키워드로 답변해 주세요.
질문: {question}
"""

META_QA_KO_PROMPT = """다음 질문에 대한 답을 알고 있나요? 답을 알고 있고 확실하다면 "예"라고만 답변해 주세요. 답을 모르거나 확실하지 않다면 "아니요"라고만 답변해 주세요.
질문: {question}
"""

DIRECT_QA_CN_PROMPT = """请用关键词回答以下问题。
问题: {question}
"""

META_QA_CN_PROMPT = """你知道以下问题的答案吗？如果你知道并确定答案，请仅回答“是”。如果你不知道或不确定，请仅回答“否”。
问题: {question}
"""

DIRECT_QA_ES_PROMPT = """Responde a la siguiente pregunta con palabras clave.
Pregunta: {question}
"""

META_QA_ES_PROMPT = """¿Conoces la respuesta a la siguiente pregunta? Si la conoces y estás seguro de ella, responde únicamente "Sí". Si no la conoces o no estás seguro, responde únicamente "No".
Pregunta: {question}
"""

DIRECT_QA_WITH_IDW_PROMPT = """Answer the following question with keywords. If you don't know the answer, just return "I don't know".
Question: {question}
"""
