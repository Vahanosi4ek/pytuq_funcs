from nodes import *
from lexer import *
from tree_manip import *

def optimize(ast, _pass_no=0):
	new = None

	while True:
		new = optimize_e_log(ast)
		new = optimize_mul_zero(new)
		new = optimize_add_zero(new)
		new = optimize_mul_one(new)
		_pass_no += 1

		if new == ast:
			break

		ast = new

	if _pass_no > 1:
		print(f"Finished optimization with {_pass_no} passes")
	else:
		print(f"Finished optimization with {_pass_no} pass")

	return new

def optimize_e_log(ast):
	if isinstance(ast, UnaryOp) and ast.op_tok.type == Tokens.Func and ast.op_tok.val == "exp" and is_c_log(ast.node):
		b, c = get_bc_c_log(ast.node).b, get_bc_c_log(ast.node).c
		return SubSuperScriptNode(b, SuffixNode(None, c))

	else:
		return apply_func_to_ast(optimize_e_log, ast)

def optimize_mul_zero(ast):
	"""
	Patterns:
	0 * ...
	... * 0
	0 / ...
	"""
	if isinstance(ast, UnaryOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub) and isinstance(ast.node, NumNode) and ast.node.tok.val == 0:
			return NumNode(Token(Tokens.Num, 0))
	elif isinstance(ast, BinOp):
		if ast.op_tok.type == Tokens.Mul:
			if isinstance(ast.left_node, NumNode) and ast.left_node.tok.val == 0:
				return NumNode(Token(Tokens.Num, 0))
			elif isinstance(ast.right_node, NumNode) and ast.right_node.tok.val == 0:
				return NumNode(Token(Tokens.Num, 0))
		elif ast.op_tok.type == Tokens.Div:
			if isinstance(ast.left_node, NumNode) and ast.left_node.tok.val == 0:
				return NumNode(Token(Tokens.Num, 0))

	return apply_func_to_ast(optimize_mul_zero, ast)

def optimize_add_zero(ast):
	"""
	Patterns:
	0 + ...
	... + 0
	... - 0
	0 - ...
	"""
	if isinstance(ast, BinOp):
		if ast.op_tok.type == Tokens.Add:
			if isinstance(ast.left_node, NumNode) and ast.left_node.tok.val == 0:
				return ast.right_node
			elif isinstance(ast.right_node, NumNode) and ast.right_node.tok.val == 0:
				return ast.left_node
		elif ast.op_tok.type == Tokens.Sub:
			if isinstance(ast.left_node, NumNode) and ast.left_node.tok.val == 0:
				return UnaryOp(Token(Tokens.Sub), ast.right_node)
			elif isinstance(ast.right_node, NumNode) and ast.right_node.tok.val == 0:
				return ast.left_node

	return apply_func_to_ast(optimize_add_zero, ast)

def optimize_mul_one(ast):
	"""
	Patterns:
	-/+1 * ...
	... * -/+1
	... / -/+1
	"""
	def is_one_or_neg_one(ast, is_neg=False):
		if isinstance(ast, NumNode) and ast.tok.val == 1:
			return True, is_neg
		elif isinstance(ast, UnaryOp) and ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			if ast.op_tok.type == Tokens.Add:
				return is_one_or_neg_one(ast.node, is_neg=is_neg)
			elif ast.op_tok.type == Tokens.Sub:
				return is_one_or_neg_one(ast.node, is_neg=not is_neg)
		return False, None

	if isinstance(ast, BinOp):
		left_is_one, left_is_neg = is_one_or_neg_one(ast.left_node)
		right_is_one, right_is_neg = is_one_or_neg_one(ast.right_node)
		if ast.op_tok.type == Tokens.Mul:
			if left_is_one:
				if left_is_neg:
					return UnaryOp(Token(Tokens.Sub), ast.right_node)
				else:
					return ast.right_node
			elif right_is_one:
				if right_is_neg:
					return UnaryOp(Token(Tokens.Sub), ast.left_node)
				else:
					return ast.left_node
		elif ast.op_tok.type == Tokens.Div:
			if right_is_one:
				if right_is_neg:
					return UnaryOp(Token(Tokens.Sub), ast.left_node)
				else:
					return ast.left_node

	return apply_func_to_ast(optimize_mul_one, ast)

# functions like a c struct
class CLogB:
	def __init__(self, b=None, c=None):
		self.b = b
		self.c = c

	def __repr__(self):
		return f"CLogB(b={self.b}, c={self.c})"

	__str__ = __repr__

	def __bool__(self):
		return self.b is not None or self.c is not None

	def set_b(self, b=None):
		self.b = b
		return self

	def set_c(self, c=None):
		self.c = c
		return self

# Returns whether ast can be represented as c log (b)
def is_c_log(ast):
	"""
	Patterns:
	(-/+)clog(b)
	(...) * clog(b)
	clog(b) * (...)
	clog(b) / (...)
	"""
	if isinstance(ast, UnaryOp) and ast.op_tok.type == Tokens.Func and ast.op_tok.val == "log":
		return True
	elif isinstance(ast, UnaryOp) and ast.op_tok.type in (Tokens.Add, Tokens.Sub):
		return is_c_log(ast.node)

	elif isinstance(ast, BinOp):
		if ast.op_tok.type == Tokens.Mul:
			if isinstance(ast.left_node, UnaryOp) and ast.left_node.op_tok.type == Tokens.Func and ast.left_node.op_tok.val == "log":
				return True
			elif isinstance(ast.right_node, UnaryOp) and ast.right_node.op_tok.type == Tokens.Func and ast.right_node.op_tok.val == "log":
				return True
			else:
				return is_c_log(ast.left_node) or is_c_log(ast.right_node)

		elif ast.op_tok.type == Tokens.Div:
			if isinstance(ast.left_node, UnaryOp) and ast.left_node.op_tok.type == Tokens.Func and ast.left_node.op_tok.val == "log":
				return True
			else:
				return is_c_log(ast.left_node)

	return False

# Returns b and c, assuming ast is c_log
def get_bc_c_log(ast):
	"""
	Patterns:
	(-/+)clog(b):		c = (-/+)get_c
	(...) * clog(b)		c = (...)get_c
	clog(b) * (...)		c = (...)get_c
	clog(b) / (...)		c = get_c / (...)
	"""
	if isinstance(ast, UnaryOp) and ast.op_tok.type == Tokens.Func and ast.op_tok.val == "log":
		return CLogB(b=ast.node, c=NumNode(Token(Tokens.Num, 1)))
	elif isinstance(ast, UnaryOp) and ast.op_tok.type in (Tokens.Add, Tokens.Sub):
		return CLogB(b=get_bc_c_log(ast.node).b, c=UnaryOp(ast.op_tok, get_bc_c_log(ast.node).c))

	elif isinstance(ast, BinOp):
		if ast.op_tok.type == Tokens.Mul:
			if isinstance(ast.left_node, UnaryOp) and ast.left_node.op_tok.type == Tokens.Func and ast.left_node.op_tok.val == "log":
				return CLogB(b=get_bc_c_log(ast.left_node.node).b, c=ast.right_node)
			elif isinstance(ast.right_node, UnaryOp) and ast.right_node.op_tok.type == Tokens.Func and ast.right_node.op_tok.val == "log":
				return CLogB(b=get_bc_c_log(ast.right_node.node).b, c=ast.left_node)
			else:
				c = BinOp(get_bc_c_log(ast.left_node).c, Token(Tokens.Mul), get_bc_c_log(ast.right_node).c)
				if is_c_log(ast.left_node):
					return CLogB(b=get_bc_c_log(ast.left_node).b, c=c)
				elif is_c_log(ast.right_node):
					return CLogB(b=get_bc_c_log(ast.right_node).b, c=c)

		elif ast.op_tok.type == Tokens.Div:
			if isinstance(ast.left_node, UnaryOp) and ast.left_node.op_tok.type == Tokens.Func and ast.left_node.op_tok.val == "log":
				return CLogB(b=get_bc_c_log(ast.left_node).b, c=ast.right_node)
			else:
				c = BinOp(get_bc_c_log(ast.left_node).c, Token(Tokens.Div), get_bc_c_log(ast.right_node).c)
				return CLogB(b=get_bc_c_log(ast.left_node).b, c=c)

	return CLogB(b=ast, c=ast)
