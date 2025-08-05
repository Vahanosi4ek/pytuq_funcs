from nodes import *
from codegen import *
from lexer import *
from parser import *
from tree_manip import *

# Takes every superscript and turns it into e**log(...) to avoid calculating the gradient of exponents
def configure_exponents(ast):
	if isinstance(ast, SubSuperScriptNode):
		if ast.suffix_node.super_ is not None:
			if ast.suffix_node.sub is None:
				return UnaryOp(Token(Tokens.Func, "exp"), BinOp(configure_exponents(ast.suffix_node.super_), Token(Tokens.Mul), UnaryOp(Token(Tokens.Func, "log"), configure_exponents(ast.node))))
			else:
				return UnaryOp(Token(Tokens.Func, "exp"), BinOp(configure_exponents(ast.suffix_node.super_), Token(Tokens.Mul), UnaryOp(Token(Tokens.Func, "log"), SubSuperScriptNode(configure_exponents(ast.node), suffix_node=configure_exponents(ast.suffix_node)))))

		else:
			return SubSuperScriptNode(configure_exponents(ast.node), suffix_node=configure_exponents(ast.suffix_node))

	elif isinstance(ast, SuffixNode):
		return SuffixNode(ast.sub, None)

	else:
		return apply_func_to_ast(configure_exponents, ast)

# Configures ast to be with respect to x_{respect_no}
def configure_ast_for_partial_derivative(ast, respect_no):
	if isinstance(ast, SubSuperScriptNode) and isinstance(ast.node, VarNode) and ast.node.tok.val == "x" and ast.suffix_node.sub is not None and int(ast.suffix_node.sub.tok.val) == respect_no:
		if ast.suffix_node.super_:
			return SubSuperScriptNode(ast.node, suffix_node=SuffixNode(super_=ast.suffix_node.super_))
		else:
			return ast.node
	else:
		return apply_func_to_ast(configure_ast_for_partial_derivative, ast, respect_no)

# Redoes the subscripting
def unconfigure_ast_for_partial_derivative(ast, respect_no, _subscripted=False):
	if isinstance(ast, SubSuperScriptNode) and isinstance(ast.node, VarNode) and ast.node.tok.val == "x" and ast.suffix_node.sub is None:
		return SubSuperScriptNode(ast.node, SuffixNode(sub=NumNode(Token(Tokens.Num, respect_no)), super_=unconfigure_ast_for_partial_derivative(ast.suffix_node.super_, respect_no)))		
	elif isinstance(ast, SubSuperScriptNode) and isinstance(ast.node, VarNode) and ast.node.tok.val == "x" and ast.suffix_node.sub is not None:
		return SubSuperScriptNode(unconfigure_ast_for_partial_derivative(ast.node, respect_no, _subscripted=True), SuffixNode(sub=ast.suffix_node.sub, super_=unconfigure_ast_for_partial_derivative(ast.suffix_node.super_, respect_no)))

	elif not _subscripted and isinstance(ast, VarNode) and ast.tok.val == "x":
		return SubSuperScriptNode(ast, SuffixNode(sub=NumNode(Token(Tokens.Num, respect_no))))

	else:
		return apply_func_to_ast(unconfigure_ast_for_partial_derivative, ast, respect_no)


def ast_to_grad_ast_1d(ast):
	if isinstance(ast, NumNode):
		return NumNode(Token(Tokens.Num, 0))

	elif isinstance(ast, VarNode):
		if ast.tok.val == "x":
			return NumNode(Token(Tokens.Num, 1))

		else:
			return NumNode(Token(Tokens.Num, 0))

	elif isinstance(ast, UnaryOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			return UnaryOp(ast.op_tok, ast_to_grad_ast_1d(ast.node))

		elif ast.op_tok.type == Tokens.Func:
			func = ast.op_tok
			arg = ast.node

			if func.val == "cos":
				return BinOp(UnaryOp(Token(Tokens.Sub), UnaryOp(Token(Tokens.Func, "sin"), arg)), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

			elif func.val == "sin":
				return BinOp((UnaryOp(Token(Tokens.Func, "cos"), arg)), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

			elif func.val == "exp":
				return BinOp((UnaryOp(Token(Tokens.Func, "exp"), arg)), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

			elif func.val == "log":
				return BinOp(BinOp(NumNode(Token(Tokens.Num, 1)), Token(Tokens.Div), arg), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

			elif func.val == "sqrt":
				return BinOp(BinOp(NumNode(Token(Tokens.Num, 1)), Token(Tokens.Div), BinOp(NumNode(Token(Tokens.Num, 2)), Token(Tokens.Mul), UnaryOp(Token(Tokens.Func, "sqrt"), VarNode(Token(Tokens.Var, "x"))))), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

			elif func.val == "abs":
				return BinOp((UnaryOp(Token(Tokens.Func, "sign"), arg)), Token(Tokens.Mul), ast_to_grad_ast_1d(arg))

	elif isinstance(ast, BinOp):
		left_grad = ast_to_grad_ast_1d(ast.left_node)
		right_grad = ast_to_grad_ast_1d(ast.right_node)

		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			return BinOp(left_grad, ast.op_tok, right_grad)

		elif ast.op_tok.type == Tokens.Mul:
			return BinOp(BinOp(left_grad, Token(Tokens.Mul), ast.right_node), Token(Tokens.Add), BinOp(ast.left_node, Token(Tokens.Mul), right_grad))

		elif ast.op_tok.type == Tokens.Div:
			return BinOp(BinOp(BinOp(left_grad, Token(Tokens.Mul), ast.right_node), Token(Tokens.Sub), BinOp(ast.left_node, Token(Tokens.Mul), right_grad)), Token(Tokens.Div), SubSuperScriptNode(ast.right_node, SuffixNode(super_=NumNode(Token(Tokens.Num, 2)))))

	elif isinstance(ast, SubSuperScriptNode):
		sub = ast.suffix_node.sub
		power = ast.suffix_node.super_
		node = ast.node

		if isinstance(node, VarNode):
			if sub:
				return NumNode(Token(Tokens.Num, 0))

			else:
				print("If this gets printed, there is an error with the grad calculator")

def ast_to_grad_ast_nd(ast, dims):
	res = []

	for d in range(dims):
		configured_ast = configure_ast_for_partial_derivative(ast, d + 1)
		res.append(unconfigure_ast_for_partial_derivative(ast_to_grad_ast_1d(configured_ast), d + 1))

	return res

def autograd_1d(latex):
	lexer = Lexer(latex)
	tokens = lexer.get_tokens()

	parser = Parser(tokens)
	ast = parser.parse()
	configured_ast = configure_exponents(ast)
	grad_ast = ast_to_grad_ast_1d(configured_ast)
	grad_ast = optimize(grad_ast)

	code = ast_to_code(grad_ast)
	return code

def autograd_nd(latex, d):
	lexer = Lexer(latex)
	tokens = lexer.get_tokens()

	parser = Parser(tokens)
	ast = parser.parse()
	configured_ast = configure_exponents(ast)
	grad_ast = ast_to_grad_ast_nd(configured_ast, d)
	res = []

	for i in grad_ast:
		optim = optimize(i)
		res.append(ast_to_code(optim))

	return res
