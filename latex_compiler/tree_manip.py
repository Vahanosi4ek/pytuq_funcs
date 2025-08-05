from nodes import *

def apply_func_to_ast(func, ast, *args, **kwargs):
	if isinstance(ast, (NumNode, VarNode)):
		return ast
	elif isinstance(ast, SubSuperScriptNode):
		return SubSuperScriptNode(func(ast.node, *args, **kwargs), suffix_node=func(ast.suffix_node, *args, **kwargs))
	elif isinstance(ast, SuffixNode):
		return SuffixNode(sub=func(ast.sub, *args, **kwargs), super_=func(ast.super_, *args, **kwargs))
	elif isinstance(ast, UnaryOp):
		return UnaryOp(ast.op_tok, func(ast.node, *args, **kwargs))
	elif isinstance(ast, BinOp):
		return BinOp(func(ast.left_node, *args, **kwargs), ast.op_tok, func(ast.right_node, *args, **kwargs))
	else:
		return None