from lexer import *
from parser import *
from codegen import *
from autograd import *

latex = r"(x^3)^2"

code = codegen(latex)
# grad = autograd_1d(latex)

print(f"code: {code}")
# print(f"grad: {grad}")