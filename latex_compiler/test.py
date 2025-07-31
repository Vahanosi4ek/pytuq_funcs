from lexer import *
from parser import *

latex = r"2 + 5 * 3"
lexer = Lexer(latex)
tokens = lexer.get_tokens()

parser = Parser(tokens)
ast = parser.parse()

print(ast)