from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from typing import Annotated, List, Any


repl = PythonREPL()


@tool
def run_python_code(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def eval_python_code(
    code: Annotated[str, "The python code to execute to generate your chart."]
) -> Any:
    """Use this to return an object in python. This returns the object required
    Use this if you are asked to return some intermediate results"""
    try:
        obj = eval(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return obj
