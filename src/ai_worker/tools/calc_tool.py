"""
ASI v8 Calc Tool
Math expression evaluation via asteval.Interpreter ONLY.
Python's built-in eval is never used.
"""

from pydantic import BaseModel, Field

from ..errors import ASIToolError
from .registry import Permission, ToolDefinition


class CalcInput(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate", max_length=2000)


class CalcOutput(BaseModel):
    expression: str
    result: str


class CalcTool(ToolDefinition):
    name = "calc"
    description = "Evaluate a mathematical expression safely using asteval."
    input_schema = CalcInput
    output_schema = CalcOutput
    required_permissions = [Permission.CALC]
    timeout_seconds = 5.0

    async def _execute(self, validated_input: CalcInput) -> CalcOutput:
        try:
            from asteval import Interpreter as ASTInterpreter
        except ImportError as exc:
            raise ASIToolError(
                "asteval library is not installed. Run: pip install asteval"
            ) from exc

        interp = ASTInterpreter()
        result = interp(validated_input.expression)

        if interp.error:
            error_msgs = "; ".join(str(e.get_error()) for e in interp.error)
            raise ASIToolError(f"Expression evaluation error: {error_msgs}")

        return CalcOutput(
            expression=validated_input.expression,
            result=str(result),
        )
