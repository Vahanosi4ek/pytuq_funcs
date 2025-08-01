from lexer import *
from parser import *
from codegen import *
from autograd import *
from optimizer import *
from function_creator import *

import numpy as np

# latex = r"x_1^2 - c_1 x_1 + c_2 \sin(c_3 \pi x_1) + c_4 \cos(c_5 \pi x_1) + c_6 - \frac{c_7}{\exp(c_8 (x_2 -c_9)^{2})}"

# code = codegen(latex)
# grad = autograd_nd(latex, 2)

# create_class("TestClass", [12.0, 8.0, 2.5, 10.0, 0.5, 11.0, 0.2 * np.sqrt(5), 0.5, 0.5], 2, 1,
	# [[-30.0, 30.0], [-30.0, 30.0]], "Test class", code, grad)

latex = r"e^{\log(x)}"

lexer = Lexer(latex)
parser = Parser(lexer.get_tokens())
ast = parser.parse()
code = ast_to_code(ast)
optim_ast = optimize(ast)
print(optim_ast)