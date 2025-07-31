class UnaryOp:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

	def __repr__(self):
		return f"UnaryOp({self.op_tok}, {self.node})"

	__str__ = __repr__

class BinOp:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

	def __repr__(self):
		return f"BinOp({self.left_node}, {self.op_tok}, {self.right_node})"

	__str__ = __repr__

class NumNode:
	def __init__(self, tok):
		self.tok = tok
		self.tok.val = float(self.tok.val)

		if self.tok.val % 1 == 0:
			self.tok.val = int(self.tok.val)

	def __repr__(self):
		return f"NumNode({self.tok})"

	__str__ = __repr__

class VarNode:
	def __init__(self, tok):
		self.tok = tok

	def __repr__(self):
		return f"VarNode({self.tok})"

	__str__ = __repr__

class SuffixNode:
	def __init__(self, sub=None, super_=None):
		self.sub = sub
		self.super_ = super_

	def __repr__(self):
		return f"SuffixNode({self.sub}, {self.super_})"

	__str__ = __repr__

class SubSuperScriptNode:
	def __init__(self, tok, suffix_node):
		self.tok = tok
		self.suffix_node = suffix_node

	def __repr__(self):
		return f"SubSuperScriptNode({self.tok}, {self.suffix_node.sub}, {self.suffix_node.super_})"

	__str__ = __repr__