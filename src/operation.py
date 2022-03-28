import operator


def safe_div(a, b):
    return a / b if abs(b) >= 1e-14 else a


ADD = {
    "func": operator.add,
    "arg_count": 2,
    "format_str": lambda a, b: f"({a} + {b})",
}
SUB = {
    "func": operator.sub,
    "arg_count": 2,
    "format_str": lambda a, b: f"({a} - {b})",
}
MUL = {
    "func": operator.mul,
    "arg_count": 2,
    "format_str": lambda a, b: f"({a} * {b})",
}
DIV = {"func": safe_div, "arg_count": 2, "format_str": lambda a, b: f"({a} / {b})"}
NEG = {"func": operator.neg, "arg_count": 1, "format_str": lambda a: f"-({a})"}
