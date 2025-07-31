from lexer import *
from parser import *

# latex = r"(x_3^2 + 5) * 2"
latex = r"x^{4+2}_{123456}"
lexer = Lexer(latex)
tokens = lexer.get_tokens()

parser = Parser(tokens)
ast = parser.parse()

print(ast)