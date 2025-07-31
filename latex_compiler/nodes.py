class SuffixNode:
	def __init__(self, sub=None, super_=None):
		self.sub = sub
		self.super_ = super_

	def __repr__(self):
		return f"SuffixNode({self.sub}, {self.super_})"

	def __bool__(self):
		return bool(self.sub or self.super_)

	__str__ = __repr__

class Node:
	def set_suffix(self, sub=None, super_=None):
		if sub is not None:
			self.suffix.sub = sub
		if super_ is not None:
			self.suffix.super_ = super_

		return self

	def reset_suffix(self, sub=False, super_=False):
		if sub:
			self.suffix.sub = None
		if super_:
			self.suffix.super_ = None

		return self

class UnaryOp(Node):
	def __init__(self, op_tok, node, suffix=SuffixNode()):
		self.op_tok = op_tok
		self.node = node
		self.suffix = suffix

	def __repr__(self):
		if self.suffix:
			return f"UnaryOp({self.op_tok}, {self.node}, {self.suffix})"
		else:
			return f"UnaryOp({self.op_tok}, {self.node})"

	__str__ = __repr__

class BinOp(Node):
	def __init__(self, left_node, op_tok, right_node, suffix=SuffixNode()):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node
		self.suffix = suffix

	def __repr__(self):
		if self.suffix:
			return f"BinOp({self.left_node}, {self.op_tok}, {self.right_node}, {self.suffix})"
		else:
			return f"BinOp({self.left_node}, {self.op_tok}, {self.right_node})"

	__str__ = __repr__

class NumNode(Node):
	def __init__(self, tok, suffix=SuffixNode()):
		self.tok = tok
		self.tok.val = float(self.tok.val)
		self.suffix = suffix

		if self.tok.val % 1 == 0:
			self.tok.val = int(self.tok.val)

	def __repr__(self):
		if self.suffix:
			return f"NumNode({self.tok}, {self.suffix})"
		else:
			return f"NumNode({self.tok})"

	__str__ = __repr__

class VarNode(Node):
	def __init__(self, tok, suffix=SuffixNode()):
		self.tok = tok
		self.suffix = suffix

	def __repr__(self):
		if self.suffix:
			return f"VarNode({self.tok}, {self.suffix})"
		else:
			return f"VarNode({self.tok})"

	__str__ = __repr__
