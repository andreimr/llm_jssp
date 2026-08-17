# Import things that are needed generically
from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
# Import things that are needed for this specific tool
from ortools.constraint_solver import pywrapcp


@tool
def reverse_string_tool(s:str) -> str:
    return s[::-1]


class JSSPInput(BaseModel):
    machines_count: int                 = Field(description="Number of machines")
    jobs_count: int                     = Field(description="Number of jobs")
    machines: list[list[int]]           = Field(description="Machines")
    processing_times: list[list[int]]   = Field(description="Processing times")

class JSSPTool(BaseTool):
    def run(self, input_data: str) -> str:
        # Implement your tool logic here
        pass
