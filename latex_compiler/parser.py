from lexer import *
from nodes import *

# THIS CODE DOES NOT DO PROPER ERROR CHECKING. ENSURE THAT THE LATEX CODE IS VALID!

# Thanks to https://github.com/davidcallanan/py-myopl-code for motivation how to start the parser
class Parser:
	def __init__(self, tokens):
		if len(tokens) == 0:
			raise Exception("Cannot pass an empty token list into a parser")

		self.tokens = tokens
		self.cur_i = -1
		self.cur = self.tokens[0]

		self.advance()

	def advance(self):
		self.cur_i += 1

		if self.cur.type != Tokens.EOF:
			self.cur = self.tokens[self.cur_i]

	def parse(self):
		return self.add_sub()

	def tiny(self):
		tok = self.cur
		if tok.type == Tokens.Num:
			self.advance()
			return NumNode(tok)

		if tok.type in (Tokens.Var, Tokens.Const):
			self.advance()
			return VarNode(tok)

		if tok.type == Tokens.Lparen:
			self.advance()
			res = self.add_sub()
			self.advance()
			return res
		
		if tok.type == Tokens.Func:
			if tok.val == "frac":
				self.advance()
				self.advance()
				num = self.add_sub()
				self.advance()
				self.advance()
				den = self.add_sub()
				self.advance()
				return BinOp(num, Token(Tokens.Div), den)
			else:
				self.advance() # go to the left paren or sub/super script
				suf = None
				if self.cur.type in (Tokens.SubScript, Tokens.SuperScript):
					suf = self.suffix()
				self.advance() # go to start of arg
				arg = self.add_sub()
				self.advance() # go past the right paren
				if suf:
					return UnaryOp(tok, arg, suf)
				else:
					return UnaryOp(tok, arg)

		raise Exception(f"Unable to parse the tokens {self.tokens} at position {self.cur_i}")

	def suffix(self):
		sub = None
		super_ = None

		if self.cur.type == Tokens.SubScript:
			# char
			if self.cur.val:
				sub = NumNode(Token(Tokens.Num, self.cur.val))
			else:
				self.advance() # go to the left brace
				self.advance() # go to the first char
				sub = self.add_sub()
			self.advance()
			if self.cur.type == Tokens.SuperScript:
				# char
				if self.cur.val:
					super_ = NumNode(Token(Tokens.Num, self.cur.val))
				else:
					self.advance()
					self.advance()
					super_ = self.add_sub()
				self.advance()

		elif self.cur.type == Tokens.SuperScript:
			# char
			if self.cur.val:
				super_ = NumNode(Token(Tokens.Num, self.cur.val))
			else:
				self.advance() # go to the left brace
				self.advance() # go to the first char
				super_ = self.add_sub()
			self.advance()
			if self.cur.type == Tokens.SubScript:
				# char
				if self.cur.val:
					sub = NumNode(Token(Tokens.Num, self.cur.val))
				else:
					self.advance()
					self.advance()
					sub = self.add_sub()
				self.advance()

		if sub or super_:
			return SuffixNode(sub, super_)
		else:
			return None

	def tiny_suffix(self):
		left = self.tiny()
		suffix = self.suffix()

		if suffix:
			left.suffix = suffix
		return left

	def simple(self):
		if self.cur.type in (Tokens.Add, Tokens.Sub):
			op = self.cur
			self.advance()
			return UnaryOp(op, self.simple())

		return self.tiny_suffix()

		raise Exception(f"Unable to parse the tokens {self.tokens} at position {self.cur_i}")

	def mul_div(self):
		left = self.simple()

		while True:
			if self.cur.type in (Tokens.Mul, Tokens.Div):
				op = self.cur
				self.advance()
				right = self.simple()
			elif self.cur.type in (Tokens.Num, Tokens.Var, Tokens.Lparen, Tokens.Const, Tokens.Func):
				op = Token(Tokens.Mul)
				right = self.tiny_suffix()
			else:
				break
			left = BinOp(left, op, right)

		return left


	def add_sub(self):
		left = self.mul_div()

		while self.cur.type in (Tokens.Add, Tokens.Sub):
			op = self.cur
			self.advance()
			right = self.mul_div()
			left = BinOp(left, op, right)

		return left
