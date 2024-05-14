import langchain_core.output_parsers
import langchain_core.prompts
import pytest
import sys
import os
from dotenv import load_dotenv

sys.path.append('../llm_jssp')
import llm_jssp
import pytest
import langchain_core
import langchain_cli 
import langchain_community
from langchain_openai import ChatOpenAI

enable_expensive_tests = False

@pytest.fixture
def generate_chat_model() -> ChatOpenAI:
    print("Setting up the OpenAI chat model. If this fails, check your API key. Also, you may not wish to be billed for using the API")
    # Generate the model using langchain
    load_dotenv()
    secret_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=secret_key)
    yield model
    # Clean up resources, if necessary

@pytest.mark.skipif(enable_expensive_tests == False, reason="This test is expensive to run")
def test_chat_model(generate_chat_model):
    model = generate_chat_model # this actually returns the model.

    prompt = langchain_core.prompts.ChatPromptTemplate.from_template("Tell me a short joke about {topic}.")
    frederick_the_string_parser = langchain_core.output_parsers.StrOutputParser()
    chain = prompt | model | frederick_the_string_parser
    response = chain.invoke({"topic": "chickens"})  # This is the input to the model
    print("Response: " + response)
    assert len(response) > 0

if __name__ == "__main__":
    pytest.main(['-s'])
