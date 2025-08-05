class UnaryOp:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

	def __repr__(self):
		return f"UnaryOp({self.op_tok}, {self.node})"

	__str__ = __repr__

	def __eq__(self, b):
		if isinstance(b, UnaryOp):
			return b.op_tok == self.op_tok and b.node == self.node
		return False

class BinOp:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

	def __repr__(self):
		return f"BinOp({self.left_node}, {self.op_tok}, {self.right_node})"

	__str__ = __repr__

	def __eq__(self, b):
		if isinstance(b, BinOp):
			return b.op_tok == self.op_tok and b.left_node == self.left_node and b.right_node == self.right_node
		return False

class NumNode:
	def __init__(self, tok):
		self.tok = tok
		self.tok.val = int(float(self.tok.val))

		assert (self.tok.val >= 0)

	def __repr__(self):
		return f"NumNode({self.tok})"

	__str__ = __repr__

	def __eq__(self, b):
		if isinstance(b, NumNode):
			return int(b.tok.val) == int(self.tok.val)
		return False

class VarNode:
	def __init__(self, tok):
		self.tok = tok

	def __repr__(self):
		return f"VarNode({self.tok})"

	__str__ = __repr__

	def __eq__(self, b):
		if isinstance(b, VarNode):
			return b.tok.val == self.tok.val
		return False

class SuffixNode:
	def __init__(self, sub=None, super_=None):
		self.sub = sub
		self.super_ = super_

	def __repr__(self):
		return f"SuffixNode({self.sub}, {self.super_})"

	def __bool__(self):
		return bool(self.sub or self.super_)

	__str__ = __repr__

	def __eq__(self, b):
		if isinstance(b, SuffixNode):
			return self.sub == b.sub and self.super_ == b.super_
		return False

class SubSuperScriptNode:
	def __init__(self, node, suffix_node=SuffixNode()):
		self.node = node
		self.suffix_node = suffix_node

	def __repr__(self):
		return f"SubSuperScriptNode({self.node}, {self.suffix_node})"

	def __eq__(self, b):
		if isinstance(b, SubSuperScriptNode):
			return self.node == b.node and self.suffix_node == b.suffix_node
		return False
