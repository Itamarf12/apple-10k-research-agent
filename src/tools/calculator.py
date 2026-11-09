"""Calculator tool: exact arithmetic so the LLM never has to do math in its head."""
import ast
import logging
import operator

logger = logging.getLogger(__name__)

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    """Evaluate a parsed expression. Only numbers, + - * / and parentheses are allowed."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers, + - * / and parentheses are allowed")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '34550 / 416161 * 100'."""
    cleaned = expression.replace(",", "").replace("$", "").replace("×", "*").replace("÷", "/").strip()
    try:
        result = _eval(ast.parse(cleaned, mode="eval"))
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except (SyntaxError, ValueError) as exc:
        logger.warning("Calculator rejected %r: %s", expression, exc)
        return f"ERROR: cannot evaluate '{expression}' ({exc})"
    return f"{expression.strip()} = {result:.4f}"
