from nodes import *
from lexer import *
from parser import *

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

def parenify(node):
	if not isinstance(node, (NumNode, VarNode)):
		return f"({ast_to_code(node)})"
	else:
		return f"{ast_to_code(node)}"

def ast_to_code(ast):
	code = ""

	if isinstance(ast, NumNode):
		code += f"{ast.tok.val}"

	elif isinstance(ast, SubSuperScriptNode):
		if ast.suffix_node.sub:
			if isinstance(ast.node, VarNode):
				if ast.node.tok.val == "x":
					code += parenify(ast.node) + f"[:, {int(ast.suffix_node.sub.tok.val) - 1}]"

				elif ast.node.tok.val == "c":
					code += "self." + parenify(ast.node) + str(ast.suffix_node.sub.tok.val)
		if ast.suffix_node.super_:
			if not ast.suffix_node.sub:
				code += f"{parenify(ast.node)}**{parenify(ast.suffix_node.super_)}"
			else:
				code += f"**{parenify(ast.suffix_node.super_)}"

	elif isinstance(ast, VarNode):
		if ast.tok.type == Tokens.Const:
			code += const_dict[ast.tok.val]
		else:
			code += ast.tok.val

	elif isinstance(ast, UnaryOp):
		if ast.op_tok.type in (Tokens.Add, Tokens.Sub):
			code += ast.op_tok.type + f"{parenify(ast.node)}"

		elif ast.op_tok.type == Tokens.Func:
			code += func_dict[ast.op_tok.val] + f"({ast_to_code(ast.node)})" # Here, we must have the parenthesis

	elif isinstance(ast, BinOp):
		code += f"{parenify(ast.left_node)}{ast.op_tok.type}{parenify(ast.right_node)}"
	
	return code

def codegen(latex):
	lexer = Lexer(latex)
	tokens = lexer.get_tokens()

	parser = Parser(tokens)
	ast = parser.parse()

	code = ast_to_code(ast)
	return code