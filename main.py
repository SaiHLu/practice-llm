from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = llm.invoke("Tell me a joke in 20 words or less.")
print(response.content)
