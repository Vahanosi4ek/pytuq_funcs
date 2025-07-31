from lexer import *
from parser import *
from codegen import *

latex = r"\frac{3xc_2}{2}"
lexer = Lexer(latex)
tokens = lexer.get_tokens()

parser = Parser(tokens)
ast = parser.parse()

code = codegen(ast)

print(code)