from codegen import *
from autograd import *

def create_class(name, consts, domain_list,
					description, latex, outdim=1):
	func_code = codegen(latex)
	dim = len(domain_list)
	grad_code_list = autograd_nd(latex, dim)

	consts1 = ""
	for i, c in enumerate(consts):
		consts1 += f"c{i + 1}={c}, "

	consts2 = ""
	for i, c in enumerate(consts):
		consts2 += f"self.c{i + 1}" + (", " if i != len(consts) - 1 else " = ")
	for i, c in enumerate(consts):
		consts2 += f"c{i + 1}" + (", " if i != len(consts) - 1 else "")

	consts3 = "("
	for i, c in enumerate(consts):
		consts3 += str(c) + (", " if i != len(consts) - 1 else ")")

	grad_str = ""
	for i, grad in enumerate(grad_code_list):
		grad_str += f"\t\tgrad[:, 0, {i}] = {grad}\n"

	code = fr'''import numpy as np
from pytuq.func.func import Function

class {name}(Function):
	"""
	{name} [https://infinity77.net/global_optimization/test_functions_nd_{name[0]}.html#go_benchmark.{name}]
	"""
	def __init__(self, {consts1}name="{name}"):
		super().__init__()
		self.name = name
		{consts2}
		self.dim = {dim}
		self.outdim = {outdim}

		self.setDimDom(domain=np.array({domain_list}))

	def __call__(self, x):
		r"""{description}

		..math::
			f(x)={latex}
'''
	if len(consts):
		code += f"\n\n\t\tDefault constant values are :math:`c = {consts3}\n"

	code += f'''
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,{dim})`.

		Returns:
			np.ndarray: Output array of size `(N,{outdim})`.
		"""
		return ({func_code}).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
{grad_str}
		return grad

'''

	return code
