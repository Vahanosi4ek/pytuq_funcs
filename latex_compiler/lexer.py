class Tokens:
	Add = "+"
	Sub = "-"
	Mul = "*"
	Div = "/"
	SubScript = "_"
	SuperScript = "^"
	Lparen = "("
	Rparen = ")"
	Lbrace = "{"
	Rbrace = "}"
	Num = "Num"
	Var = "Var"
	Func = "Func"
	Const = "Const"

	# End of function
	EOF = "EOF"

class Token:
	def __init__(self, type_, val=None):
		self.type = type_
		self.val = val

	def __repr__(self):
		return f"[ {self.type}, {self.val} ]" if self.val else f"[ {self.type} ]"

	__str__ = __repr__

class Lexer:
	def __init__(self, func_str):
		self.func_str = func_str

		self.funcs = {"cos", "sin", "log", "exp", "sum", "prod", "sign", "abs", "pow"}
		self.consts = {"pi"}

	def get_tokens(self):
		s = self.func_str
		i = 0
		tokens = []

		def read_while(cond):
			nonlocal i
			start = i
			while i < len(s) and cond(s[i]):
				i += 1
			return s[start:i]

		while i < len(s):
			c = s[i]

			if c in " \t\n":
				i += 1
				continue

			if c in "+-*/":
				tokens.append(Token(c))
				i += 1
				continue

			if c == "(":
				tokens.append(Token(Tokens.Lparen))
				i += 1
				continue
			if c == ")":
				tokens.append(Token(Tokens.Rparen))
				i += 1
				continue

			if c == "{":
				tokens.append(Token(Tokens.Lbrace))
				i += 1
				continue
			if c == "}":
				tokens.append(Token(Tokens.Rbrace))
				i += 1
				continue

			if c == "_":
				if s[i + 1] == Tokens.Lbrace:
					tokens.append(Token(Tokens.SubScript))
					i += 1
				else:
					tokens.append(Token(Tokens.SubScript, s[i + 1]))
					i += 2
				continue
			if c == "^":
				if s[i + 1] == Tokens.Lbrace:
					tokens.append(Token(Tokens.SuperScript))
					i += 1
				else:
					tokens.append(Token(Tokens.SuperScript, s[i + 1]))
					i += 2
				continue

			if c.isdigit() or (c == "." and i+1 < len(s) and s[i+1].isdigit()):
				num = read_while(lambda x: x.isdigit() or x == ".")
				tokens.append(Token(Tokens.Num, num))
				continue

			if c == "\\":
				i += 1
				name = read_while(lambda x: x.isalpha())
				if name in self.funcs:
					tokens.append(Token(Tokens.Func, name))
				elif name in self.consts:
					tokens.append(Token(Tokens.Const, name))
				else:
					raise Exception(f"Unknown function {name}. If it's a real numpy function or constant, add it into lexer.py in Lexer.__init__()")
				continue

			if c.isalpha():
				name = read_while(lambda x: x.isalnum())
				tokens.append(Token(Tokens.Var, name))
				continue

			raise ValueError(f"Unexpected character '{c}' at position {i}")

		tokens.append(Token(Tokens.EOF))

		return tokens
