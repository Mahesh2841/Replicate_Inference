import os
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import replicate
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_key"



# 8B parameter LLM model:

llm = Replicate(
    model="meta/meta-llama-3-8b-instruct",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
llm(prompt)



#70B parameter model:

llm = Replicate(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model="meta/meta-llama-3-70b-instruct",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
_ = llm(prompt)



#405B parameter model:

input = {
    "prompt": "Answer the following yes/no question by reasoning step by step. Can a dog drive a car?",
    "max_tokens": 500,
    "temperature": 0.75,
    "top_p": 1
}
for event in replicate.stream("meta/meta-llama-3.1-405b-instruct", input=input):
    print(event, end="")
