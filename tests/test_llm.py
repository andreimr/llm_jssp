import torch
import pytest
import openai
import sys

sys.path.append('../llm_jssp')
import llm_jssp
import pytest
import langchain


@pytest.fixture
def generate_model():
    # Generate the model using langchain
    model = "fred"
    yield model
    # Clean up resources, if necessary
