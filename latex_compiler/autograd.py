from nodes import *
from codegen import *
from lexer import *
from parser import *

def ast_to_grad_ast_1d(ast):
	if isinstance(ast, NumNode):
		return NumNode(Token(Tokens.Num, 0))

	elif isinstance(ast, VarNode):
		if ast.tok.val == "x":
			if not ast.suffix:
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
			return BinOp(BinOp(BinOp(left_grad, Token(Tokens.Mul), ast.right_node), Token(Tokens.Sub), BinOp(ast.left_node, Token(Tokens.Mul), right_grad)), Token(Tokens.Div), ast.right_node.set_suffix(super_=NumNode(Token(Tokens.Num, 2))))

def autograd_1d(latex):
	lexer = Lexer(latex)
	tokens = lexer.get_tokens()

	parser = Parser(tokens)
	ast = parser.parse()
	grad_ast = ast_to_grad_ast_1d(ast)

	code = ast_to_code(grad_ast)
	return code