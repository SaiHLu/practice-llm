from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """
You're a helpful assistant that can answer the question about the fruit only.
If you don't know, just say 'I don\'t know' and don't try to make an answer.
"""

system_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": template},
        {"role": "human", "content": "{question}"},
    ]
)


chain = system_prompt | llm
response = chain.invoke({"question": "Tell me a joke in 20 words or less."})
print("From Messages: ", response.content)


complete_prompt_template = """
You're a helpful assistant that can answer the question about the fruit only.
If you don't know, just say 'I don\'t know' and don't try to make an answer.

Question: {question}

Helpful Answer:
"""

complete_prompt = ChatPromptTemplate.from_template(template=complete_prompt_template)
complete_prompt_chain = complete_prompt | llm
response = complete_prompt_chain.invoke(
    {"question": "Tell me a joke in 20 words or less."}
)
print("From Template: ", response.content)


placeholder_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
    You're a helpful assistant that can answer the question about the fruit only.
    If you don't know, just say 'I don\'t know' and don't try to make an answer.
  """
        ),
        MessagesPlaceholder("messages"),
    ]
)

placeholder_prompt_chain = placeholder_prompt | llm
response = placeholder_prompt_chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="How's the taste of grape, answer in 20 words or less."
            )
        ]
    }
)
print("From Message Placeholder", response.content)
