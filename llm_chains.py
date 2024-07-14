from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Optional
from langchain_openai import OpenAI, ChatOpenAI

model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="<ENTER_OPENAI_KEY_HERE>", temperature=0)

class Answer(BaseModel):
    answer: str = Field(description="answer for the given query")

answer_parser = PydanticOutputParser(pydantic_object=Answer)

answer_template = """Considering only explicitly written facts in the context, provide the most suitable answer for the query. If there is no sufficient information present in the context, please return "Not enough information" as the answer.
{format_instructions}

Context: {context}

Query: {query}
Choices: {choices}
"""

answer_prompt = PromptTemplate(
    template=answer_template,
    input_variables=["query", "choices", "context"],
    partial_variables={"format_instructions": answer_parser.get_format_instructions()},
)

answer_chain = answer_prompt | model | answer_parser


application_parser = JsonOutputParser()

application_template = """Given below string of application for a commercial cyber insurance policy extract questions, choices available to answer and user provided response from it and in JSON format. Choices, answers are immediately after application question and for some questions there are no choices or no provided answers.
{format_instructions}

Application: {application}
"""

format_template = """The output should be formatted as a JSON instance that conforms to the JSON schema below.\n{"questions": [{"question" : "question from application", "choices" : [choices for answering the question], "answer" : "existing answer"}]}"""

application_prompt = PromptTemplate(
    template=application_template,
    input_variables=["application"],
    partial_variables={"format_instructions": format_template},
)

application_chain = application_prompt | model | application_parser

