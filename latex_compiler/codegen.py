from nodes import *
from lexer import *

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

def codegen(ast):
	code = ""

	if isinstance(ast, NumNode):
		code += f"{ast.tok.val}"

	elif isinstance(ast, VarNode):
		if ast.tok.val != "x":
			print(f"Warning: the variable {ast.tok.val} may be invalid.")

		code += f"{ast.tok.val}"

	elif isinstance(ast, UnaryOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			code = ast.op_tok.type + f"({codegen(ast.node)})"

		elif ast.op_tok.type == Tokens.Func:
			code = func_dict[ast.op_tok.val] + f"({codegen(ast.node)})"

	elif isinstance(ast, BinOp):
		left_code = codegen(ast.left_node)
		right_code = codegen(ast.right_node)
		if not isinstance(ast.left_node, (NumNode, VarNode, SubSuperScriptNode)):
			left_code = f"({left_code})"
		if not isinstance(ast.right_node, (NumNode, VarNode, SubSuperScriptNode)):
			right_code = f"({right_code})"
		code += f"{left_code}{ast.op_tok.type}{right_code}"

	elif isinstance(ast, FracNode):
		code += f"({codegen(ast.num)})/({codegen(ast.den)})"

	elif isinstance(ast, SubSuperScriptNode):
		if isinstance(ast.node, NumNode):
			if ast.suffix_node.sub:
				raise Exception(f"Can't subscript {ast.node.tok.val}")
			power = ast.suffix_node.super_
			code += str(ast.node.tok.val) + f"**({codegen(power)})"

		elif isinstance(ast.node, VarNode):
			sub = ast.suffix_node.sub
			power = ast.suffix_node.super_

			var = ast.node.tok.val
			if var == "x":
				if not sub:
					code += var
				else:
					code += var + f"[:, {int(sub.tok.val) - 1}]"

			if var == "c":
				if not sub:
					print(f"Warning: the variable {var} may be invalid.")
				else:
					code += "self." + var + str(sub.tok.val)

			if not sub and power:
				code += var + f"**({codegen(power)})"
			if sub and power:
				code += f"**({codegen(power)})"

	return code
