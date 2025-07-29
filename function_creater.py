# Takes latex of function and converts to python
# Very useful to save time and avoid dumb errors, but it has limitations

# Stuff to know:
# 1. custom functions in benchmark.py which are used in the Translator class should be prefixed with _
# 1.5. Also, make sure to add that func name inside the Translator __init__
# 2. THE ONLY VARIABLES ARE c_..., x_..., or d
# 2.5. Make sure the variables have spaces between them (won't work otherwise)
# 3. Can't handle multi-variable numpy functions, and for customs you must make it like _frac in benchmark.py
# 4. Can't handle summations, products, or anything involving x_i
# 5. This code has a lot of weird hacky stuff which make no sense. It is not meant to be use for the long-term, just to make my life slightly easier

import numpy as np

class LexerState:
	Other = 0
	Numerical = 1
	Constant = 2
	Special = 3

class Tokens:
	Add = "+"
	Sub = "-"
	Mul = "*"
	Div = "/"
	Pow = "^"
	Lparen = "Lparen"
	Rparen = "Rparen"
	Num = "Num"
	Var = "Var"
	Func = "Func"
	ConstFunc = "ConstFunc"

class Token:
	def __init__(self, type_, val=None):
		self.type = type_
		self.val = val

	def __repr__(self):
		return f"({self.type}, {self.val})" if self.val else f"({self.type})"

	__str__ = __repr__

class Translator:
	def __init__(self, func_str):
		self.func_str = func_str

		self.np_funcs = ("cos", "sin", "log", "pow", "exp", "abs", "sqrt", "sign")
		self.np_consts = ("pi")
		self.custom_funcs = ("frac")

	def get_tokens(self):
		s = self.func_str
		i = 0
		tokens = []

		def read_while(pred):
			nonlocal i
			start = i
			while i < len(s) and pred(s[i]):
				i += 1
			return s[start:i]

		while i < len(s):
			c = s[i]

			if c in " \t\n":
				i += 1
				continue

			if c in "+-*/^":
				tokens.append(Token(c))
				i += 1
				continue

			if c == '(':
				if len(tokens) != 0 and tokens[-1].type in (Tokens.Rparen, Tokens.Num, Tokens.ConstFunc, Tokens.Var):
					tokens.append(Token(Tokens.Mul))
				tokens.append(Token(Tokens.Lparen))
				i += 1
				continue
			if c == ')':
				tokens.append(Token(Tokens.Rparen))
				i += 1
				continue

			if c == '{':
				# Brackets aren't visible, so unlike Lparen, we can't assume multiplication
				tokens.append(Token(Tokens.Lparen, '{'))
				i += 1
				continue
			if c == '}':
				tokens.append(Token(Tokens.Rparen, '}'))
				i += 1
				continue

			if c.isdigit() or (c == '.' and i+1 < len(s) and s[i+1].isdigit()):
				if len(tokens) != 0 and tokens[-1].type in (Tokens.Rparen, Tokens.ConstFunc, Tokens.Var):
					tokens.append(Token(Tokens.Mul))
				num = read_while(lambda x: x.isdigit() or x == '.')
				tokens.append(Token(Tokens.Num, float(num)))
				continue

			if c == '\\':
				if len(tokens) != 0 and tokens[-1].type in (Tokens.Rparen, Tokens.Num, Tokens.ConstFunc, Tokens.Var):
					tokens.append(Token(Tokens.Mul))
				i += 1
				name = read_while(lambda x: x.isalpha())
				if name in self.custom_funcs:
					tokens.append(Token(Tokens.Func, "_" + name))
				elif name in self.np_funcs:
					tokens.append(Token(Tokens.Func, "np." + name))
				elif name in self.np_consts:
					tokens.append(Token(Tokens.ConstFunc, "np." + name))
				else:
					raise ValueError(f"Unknown function: {name} at position {i}")
				continue

			if c.isalpha():
				if len(tokens) != 0 and tokens[-1].type in (Tokens.Rparen, Tokens.Num, Tokens.ConstFunc, Tokens.Var):
					tokens.append(Token(Tokens.Mul))
				name = read_while(lambda x: x.isalnum() or x == '_')
				prefix = "self."
				if "c_" in name:
					name = name[:1] + name[2:]
				elif "x_" in name:
					prefix = ""
					index = int(name[2:])
					name = f"x[:, {index - 1}]"

				tokens.append(Token(Tokens.Var, prefix + name))
				continue

			raise ValueError(f"Unexpected character '{c}' at position {i}")

		return tokens

	def token_to_code(self, tokens):
		code = ""

		for token in tokens:
			if token.type in (Tokens.Add, Tokens.Sub, Tokens.Mul, Tokens.Div):
				code += token.type
			elif token.type == Tokens.Pow:
				code += "**"
			elif token.type == Tokens.Lparen:
				code += "("
			elif token.type == Tokens.Rparen:
				code += ")"
			elif token.type == Tokens.Num:
				code += str(token.val)
			elif token.type in (Tokens.Var, Tokens.Func, Tokens.ConstFunc):
				code += token.val

		return code

def infinity77_create_class(name, consts, dim, outdim, domain_str,
					description, latex):
	consts1 = ""
	for i, c in enumerate(consts):
		consts1 += f"c{i + 1}={c}, "

	consts2 = ""
	for i, c in enumerate(consts):
		consts2 += f"self.c{i + 1}" + (", " if i != len(consts) - 1 else " = ")
	for i, c in enumerate(consts):
		consts2 += f"c{i + 1}" + (", " if i != len(consts) - 1 else "")

	consts3 = "("
	for i, c in enumerate(consts):
		consts3 += str(c) + (", " if i != len(consts) - 1 else ")")

	translator = Translator(latex)
	tokens = translator.get_tokens()
	latex_code = translator.token_to_code(tokens)
	print(tokens)
	code = fr'''class {name}(Function):
	"""
	{name} [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.{name}]
	"""
	def __init__(self, {consts1}name="{name}"):
		super().__init__()
		self.name = name
		{consts2}
		self.dim = {dim}
		self.outdim = {outdim}

		self.setDimDom(domain={domain_str})

	def __call__(self, x):
		r"""{description}

		..math::
			f(x)={latex}


		Default constant values are :math:`c = {consts3}
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,{dim})`.

		Returns:
			np.ndarray: Output array of size `(N,{outdim})`.
		"""
		return ({latex_code}).reshape(-1, 1)'''

	return code

# For example
out = infinity77_create_class("Bukin6", (100., 0.01, 0.01, 10.), 2, 1, "np.array([[-15., -5.], [-3., 3.]])", "A nondifferentiable function which has many local minima", r"c_1\sqrt{\abs{x_2-c_2 x_1^2}}+c_3\abs{x_1+c_4}")