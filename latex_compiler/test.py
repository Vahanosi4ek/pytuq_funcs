from lexer import *
from parser import *
from codegen import *
from autograd import *
from optimizer import *
from function_creator import *

import numpy as np

# (name, latex, constants, domain)
func_list = [
	("Dolan", r"\abs{(x_1 + c_1x_2)\sin(x_1) - c_2x_3 - c_3x_4\cos(x_5 + x_5 - x_1) + c_4x_5^2 - x_2 - c_5}", [1.7, 1.5, 0.1, 0.2, 1], [[-100, 100],[-100, 100],[-100, 100],[-100, 100],[-100, 100]])
]

code = ""
for func in func_list:
	code += create_class(func[0], func[2], func[3], "A multimodal minimzation function", func[1])

with open("autogen.py", "w") as f:
	f.write(code)