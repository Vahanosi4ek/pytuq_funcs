from nodes import *
from lexer import *
from parser import *
from autograd import *

func_dict = {
	"sin": "np.sin",
	"cos": "np.cos",
	"exp": "np.exp",
	"log": "np.log",
	"abs": "np.abs",
	"sign": "np.sign",
	"sum": "np.sum",
	"prod": "np.prod",
}

const_dict = {
	"pi": "np.pi"
}

def ast_to_code(ast):
	code = ""

	if isinstance(ast, NumNode):
		code += f"{ast.tok.val}"
		if ast.suffix:
			if ast.suffix.sub:
				raise Exception(f"Can't subscript {ast.tok.val}")
			power = ast.suffix.super_
			code += f"**({ast_to_code(power)})"

	elif isinstance(ast, VarNode):
		sub = ast.suffix.sub
		power = ast.suffix.super_

		var = ast.tok.val
		if ast.tok.type == Tokens.Const:
			code += const_dict[ast.tok.val]

		elif var == "x":
			if not sub:
				code += var
			else:
				code += var + f"[:, {int(sub.tok.val) - 1}]"

		elif var == "c":
			if not sub:
				print(f"Warning: the variable {var} may be invalid.")
			else:
				code += "self." + var + str(sub.tok.val)

		if power:
			code += f"**({ast_to_code(power)})"

	elif isinstance(ast, UnaryOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			code += ast.op_tok.type + f"({ast_to_code(ast.node)})"

		elif ast.op_tok.type == Tokens.Func:
			if ast.suffix:
				code += func_dict[ast.op_tok.val] + f"({ast_to_code(ast.node)})**({ast_to_code(ast.suffix.super_)})"
			else:
				code += func_dict[ast.op_tok.val] + f"({ast_to_code(ast.node)})"

	elif isinstance(ast, BinOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			base = f"{ast_to_code(ast.left_node)}{ast.op_tok.type}{ast_to_code(ast.right_node)}"
		else:
			base = f"({ast_to_code(ast.left_node)}){ast.op_tok.type}({ast_to_code(ast.right_node)})"
		if ast.suffix:
			code += f"({base})**{(ast_to_code(ast.suffix.super_))}"
		else:
			code += f"({base})"

	return code

def codegen(latex):
	lexer = Lexer(latex)
	tokens = lexer.get_tokens()

	parser = Parser(tokens)
	ast = parser.parse()

	code = ast_to_code(ast)
	print(ast)
	return configure_exponent_ast(ast)