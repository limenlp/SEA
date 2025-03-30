MULTICHOICE_QA = """# Your Task
According to the following title and context, reply with {num_of_qa} multiple-choice questions and a statement based on the question. 
- Each question should include 4 options.
- The context is only for you to generate the question. Therefore, your question should not redirect to this context, like "according to the provided context...".
- Your question should include enough information that can help the testee to understand the background.
- Your reply should follow the given JSON format.

## Title
{title}

## Context (Only you can see this context) 
{context}

## Reply format
[
    {{
        "question": "QUESTION CONTENT",
        "options": ["A: ...", "B: ...", "C: ...", "D: ..."],
        "statement" "STATEMENT OF THIS QUESTION.
        "answer": "CHOICE FROM THE OPTIONS. For example, A"
    }},
    ...
]
"""


REPHRASE_QA = """# Your Task
According to the following title, context, and a question, reply with {num_of_qa} rephrased questions and the corresponding answers. Your question should provide sufficient content to avoid ambiguous. You should reply with json format as follows:
- Your question should has the same meaning as the provided question, only rephrased.
- The context is only for you to generate the question. Therefore, your question should not redirect to this context, like "according to the provided context...".
- Your question should include enough information that can help the testee to understand the background.
- Your reply should follow the given JSON format.

## Title
{title}

## Context (Only you can see this context)
{context}

## Question
{question}

## Reply format
[
    {{
        "question": "QUESTION CONTENT",
        "options": ["A: ...", "B: ...", "C: ...", "D: ..."],
        "statement" "STATEMENT OF THIS QUESTION.
        "answer": "CHOICE FROM THE OPTIONS. For example, A"
    }},
    ...
]
"""


ANALYSIS = """# Your Task
Given a context, a question, and its corresponding incorrect solution, generate a gerund phrase that thoroughly and precisely describes the **specific** skill or capability lacking that cause the error.

## Context
{context}

## Question
{question}

## Correct Solution
{answer}

## Incorrect Solution
{llm_answer}

## Requirement
- Incorrect Solution is provided by a testee that cannot access the context. Your answer should not mention that the skill that related to the context information retrieval.
- The skill description should be an action-oriented gerund phrase that is **informative** and **detailed**.
- The phrase should refer to a **specific** skill or capability that comprehensively
covers the key aspects of the solution, without including any context or specifics from the question or solution.
- Avoid unnecessary elements unrelated to the core capability.
- Please output **only a gerund phrase** describing the skill, with NO additional text.
"""


TESTEE = """Given the topic: {topic}, answer the following question by choosing one option in Options:
Question: {que}
Options: 
{opts}
Your Answer (put your answer in \\box{{}}):
"""


TESTEE_VISION = """Answer the following question by choosing one option in Options:
Question: {que}
Options: 
{opts}
Your Answer (put your answer in \\box{{}}):
"""