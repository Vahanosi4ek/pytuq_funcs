#!/usr/bin/env python3

import numpy as np
from pytuq.func.func import Function

# TODO: add complete support for discontinuous and nondifferentiable functions

# https://infinity77.net/global_optimization/test_functions.html#test-functions-index
# 1-d functions

class SineSum(Function):
	"""
	Problem 02 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem02]
	"""
	def __init__(self, c1=1., c2=10/3, name="SineSum"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([2.7, 7.5]))

	def __call__(self, x):
		r"""Simple sum of sines

		.. math::
			f(x)=\sin(c_1x)+\sin(c_2x)


		Default constant values are :math:`c = (1., 10/3)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.sin(self.c1 * x) + np.sin(self.c2 * x)

	def grad(self, x):
		return (self.c1 * np.cos(self.c1 * x) + self.c2 * np.cos(self.c2 * x))[:, np.newaxis, :]

class SineSum2(Function):
	"""
	Problem 03 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem03]
	"""
	def __init__(self, c1=6, name="SineSum2"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""Simple sum of sines, has several local minima

		.. math::
			f(x)=-\sum_{k=1}^{c_1}k\sin((k+1)x+k)


		Default constant values are :math:`c = (6)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		summation = np.zeros((x.shape[0], self.c1))
		k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
		summation += k * np.sin((k + 1) * x + k)

		return -np.sum(summation, axis=1, keepdims=True)

	def grad(self, x):
		_ = self.__call__(x)
		summation = np.zeros((x.shape[0], self.c1))
		k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
		summation += k * (k + 1) * np.cos((k + 1) * x + k)

		return -np.sum(summation, axis=1, keepdims=True)[:, np.newaxis, :]

class QuadxExp(Function):
	"""
	Problem 04 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem04]
	"""
	def __init__(self, c1=16., c2=-24., c3=5., name="QuadxExp"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([1.9, 3.9]))

	def __call__(self, x):
		r"""Product of quadratic and exponent

		.. math::
			f(x)=-(c_1x^2+c_2x+c_3)e^{-x}


		Default constant values are :math:`c = (16., -24., 5.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._quad = self.c1 * x ** 2 + self.c2 * x + self.c3
		self._exp = np.exp(-x)
		return (-self._quad * self._exp)

	def grad(self, x):
		_ = self.__call__(x)
		return -(self._quad * -self._exp + self._exp * (2 * self.c1 * x + self.c2))[:, np.newaxis, :]

class LinxSin(Function):
	"""
	Problem 05 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem05]
	"""
	def __init__(self, c1=1.4, c2=-3., c3=18., name="LinxSin"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.2]))

	def __call__(self, x):
		r"""Product of linear and sine functions

		.. math::
			f(x)=-(c_1-c_2x)sin(c_3x)


		Default constant values are :math:`c = (1.4, -3., 18.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._linear = (self.c1 + self.c2 * x)
		self._sine = np.sin(self.c3 * x)
		return -self._linear * self._sine

	def grad(self, x):
		_ = self.__call__(x)
		return -(self._linear * self.c3 * np.cos(self.c3 * x) + self._sine * self.c2)[:, np.newaxis, :]

class SinexExp(Function):
	"""
	Problem06 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem06]
	"""
	def __init__(self, name="SinexExp"):
		super().__init__()
		self.name = name
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""Product of sine and exp functions

		.. math::
			f(x)=-(x+\sin(x))e^{-x^2}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._sine = (x + np.sin(x))
		self._exp = np.exp(-x ** 2)
		return -self._sine * self._exp

	def grad(self, x):
		_ = self.__call__(x)
		return -(self._sine * (-self._exp * 2 * x) + self._exp * (np.cos(x) + 1))[:, np.newaxis, :]

class SineLogSum(Function):
	"""
	Problem07 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem07]
	"""
	def __init__(self, c1=1., c2=10/3, c3=np.exp(1), c4=-.84, c5=3., name="SineLogSum"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([2.7, 7.5]))

	def __call__(self, x):
		r"""Sum of sine and log functions

		.. math::
			f(x)=\sin(c_1x) + \sin(c_2x) + \log_{c_3}(x) + c_4x + c_5


		Default constant values are :math:`c = (1., 10/3, e, -0.84, 3.).

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.sin(self.c1 * x) + np.sin(self.c2 * x) + np.emath.logn(self.c3, x) + self.c4 * x + self.c5

	def grad(self, x):
		return (self.c1 * np.cos(self.c1 * x) + self.c2 * np.cos(self.c2 * x) + 1 / (x * np.log(self.c3)) + self.c4)[:, np.newaxis, :]

class CosineSum(Function):
	"""
	Problem 08 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem08]
	"""
	def __init__(self, c1=6, name="CosineSum"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""Simple sum of cosines, has several local minima

		.. math::
			f(x)=-\sum_{k=1}^{c_1}k\cos((k+1)x+k)


		Default constant values are :math:`c = (6)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		summation = np.zeros((x.shape[0], self.c1))
		k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
		summation += k * np.cos((k + 1) * x + k)

		return -np.sum(summation, axis=1, keepdims=True)

	def grad(self, x):
		summation = np.zeros((x.shape[0], self.c1))
		k = np.broadcast_to(np.arange(1, self.c1 + 1), (x.shape[0], self.c1))
		summation -= k * (k + 1) * np.sin((k + 1) * x + k)

		return -np.sum(summation, axis=1, keepdims=True)[:, np.newaxis, :]

class Sinex(Function):
	"""
	Problem10 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem10]
	"""
	def __init__(self, name="Sinex"):
		super().__init__()
		self.name = name
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

	def __call__(self, x):
		r"""Product of x and sine function, has several local minima

		.. math::
			f(x)=-x\sin(x)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return -x * np.sin(x)

	def grad(self, x):
		return (-x * np.cos(x) - np.sin(x))[:, np.newaxis, :]

class CosineSum2(Function):
	"""
	Problem11 [https://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem11]
	"""
	def __init__(self, c1=2., c2=2., name="CosineSum2"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-np.pi / 2, np.pi * 2]))

	def __call__(self, x):
		r"""Simple sum of cosines

		.. math::
			f(x)=c_1\cos(x) + \cos(c_2x)


		Default constant values are :math:`c = (2., 2.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return self.c1 * np.cos(x) + np.cos(self.c2 * x)

	def grad(self, x):
		return (-self.c1 * np.sin(x) - self.c2 * np.sin(self.c2 * x))[:, np.newaxis, :]

# https://infinity77.net/global_optimization/test_functions.html#test-functions-index
# sorted alphabetically (as in the website)

class Ackley(Function):
	"""
	Ackley [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Ackley]
	"""
	def __init__(self, c1=20., c2=0.2, c3=2*np.pi, c4=20., d=2, name="Ackley"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.d = c1, c2, c3, c4, d
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-32, 32]))

	def __call__(self, x):
		r"""Complex cosine function with many local minima

		.. math::
			f(x)=-c_1e^{-c_2\sqrt{\sum_{i=1}^{d}x_1^2}}-e^{\frac{1}{d}\sum_{i=1}^{d}\cos(c_3x_i)}+c_4+e


		Default constant values are :math:`c = (20., 0.2, 2\pi)` and :math:`d = 2`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		t1 = np.exp(-self.c2 * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d))
		t2 = np.exp(np.sum(np.cos(self.c3 * x), axis=1, keepdims=True) / self.d)
		return -self.c1 * t1 - t2 + self.c4 + np.exp(1.)

	def grad(self, x):
		t1 = -np.exp(-self.c2 * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d)) * self.c2 * x / (self.d * np.sqrt(np.sum(x ** 2, axis=1, keepdims=True) / self.d))
		t2 = -np.exp(np.sum(np.cos(self.c3 * x), axis=1, keepdims=True) / self.d) * self.c3 * np.sin(self.c3 * x) / self.d

		return (-self.c1 * t1 - t2)[:, np.newaxis, :]

class Adjiman(Function):
	"""
	Adjiman [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Adjiman]
	"""
	def __init__(self, c1=1., name="Adjiman"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-1., 2.,], [-1., 1.]]))

	def __call__(self, x):
		r"""A 2d multimodal function

		.. math::
			f(x)=\cos(x_1)\sin(x_2)-\frac{x_1}{x_2^2+c_1}


		Default constant values are :math:`c = (1.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.cos(x[:, 0]) * np.sin(x[:, 1]) - x[:, 0] / (x[:, 1] ** 2 + self.c1))[:, np.newaxis]

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		x1, x2 = x[:, 0], x[:, 1]

		grad[:, 0, 0] = np.sin(x2) * -np.sin(x1) - 1 / (x2 ** 2 + self.c1)
		grad[:, 0, 1] = np.cos(x1) * np.cos(x2) - x1 * -1 / (x2 ** 2 + self.c1) ** 2 * 2 * x2

		return grad

class Alpine01(Function):
	"""
	Alpine01 [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Alpine01]
	"""
	def __init__(self, c1=0.1, d=2, name="Alpine01"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""A N-d multimodal function

		.. math::
			f(x)=\sum_{i=1}^n\abs{x_i\sin(x_i)+c_1x_2}


		Default constant values are :math:`c = (0.1)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.sum(np.abs(x * np.sin(x) + self.c1 * x), axis=1, keepdims=True)

	def grad(self, x):
		return (np.sign(x * np.sin(x) + self.c1 * x) * (x * np.cos(x) + np.sin(x) + self.c1))[:, np.newaxis, :]

class Alpine02(Function):
	"""
	Alpine02 [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Alpine02]
	"""
	def __init__(self, d=2, name="Alpine02"):
		super().__init__()
		self.name = name
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

	def __call__(self, x):
		r"""A N-d multimodal function

		.. math::
			f(x)=\prod_{i=1}^{n}\sqrt{x_i}\sin(x_i)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.prod(np.sqrt(x) * np.sin(x), axis=1, keepdims=True)

	def grad(self, x):
		inner = np.sqrt(x) * np.sin(x)
		inner_grad = np.sqrt(x) * np.cos(x) + 1 / 2 * x ** (-1 / 2) * np.sin(x)
		return (self.__call__(x) / inner * inner_grad)[:, np.newaxis, :]

class AMGM(Function):
	"""
	AMGM [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.AMGM]
	"""
	def __init__(self, d=2, name="AMGM"):
		super().__init__()
		self.name = name
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

	def __call__(self, x):
		r"""Difference between AM and GM

		.. math::
			f(x)=\left ( \frac{1}{n} \sum_{i=1}^{n} x_i - \sqrt[n]{ \prod_{i=1}^{n} x_i} \right )^2

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._am = 1 / self.dim * np.sum(x, axis=1, keepdims=True)
		self._gm = np.power(np.prod(x, axis=1, keepdims=True), 1 / self.dim)
		return (self._am - self._gm) ** 2

	def grad(self, x):
		_ = self.__call__(x)
		return (2 * (self._am - self._gm) * (1 / self.dim - 1 / self.dim * np.power((np.prod(x, axis=1, keepdims=True)), -1 / self.dim) * np.prod(x, axis=1, keepdims=True) / x))[:, np.newaxis, :]

class BartelsConn(Function):
	"""
	BartelsConn [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.BartelsConn]
	"""
	def __init__(self, name="BartelsConn"):
		super().__init__()
		self.name = name
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-50., 50.]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		.. math::
			f(x)=\abs{x_1^2+x_2^2+x_1x_2}+\abs{\sin(x_1)}+\abs{\cos(x_2)}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		self._t1 = x1 ** 2 + x2 ** 2 + x1 * x2
		self._t2 = np.sin(x1)
		self._t3 = np.cos(x2)
		return (np.abs(self._t1) + np.abs(self._t2) + np.abs(self._t3))[:, np.newaxis]

	def grad(self, x):
		_ = self.__call__(x)
		x1, x2 = x[:, 0], x[:, 1]
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = np.sign(self._t1) * (2 * x1 + x2) + np.sign(self._t2) * np.cos(x1)
		grad[:, 0, 1] = np.sign(self._t1) * (2 * x2 + x1) + np.sign(self._t3) * -np.sin(x2)
		return grad

class Bird(Function):
	"""
	Bird [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bird]
	"""
	def __init__(self, c1=1., c2=1., name="Bird"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-2 * np.pi, 2 * np.pi]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		.. math::
			f(x)=(x_1-x_2)^2+e^{(c_1-\sin(x_1))^2}\cos(x_2)+e^{(c_2-\cos(x_2))^2}\sin(x_1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.


		Default constant values are :math:`c = (1., 1.)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		self._t1 = x1 - x2
		self._t2_1 = np.exp((self.c1 - np.sin(x1)) ** 2)
		self._t2_2 = np.cos(x2)
		self._t3_1 = np.exp((self.c2 - np.cos(x2)) ** 2)
		self._t3_2 = np.sin(x1)
		return (self._t1 ** 2 + self._t2_1 * self._t2_2 + self._t3_1 * self._t3_2)[:, np.newaxis]

	def grad(self, x):
		_ = self.__call__(x)
		x1, x2 = x[:, 0], x[:, 1]
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		_t2_1_grad = self._t2_1 * 2 * (self.c1 - np.sin(x1)) * -np.cos(x1)
		_t2_2_grad = -np.sin(x2)
		_t3_1_grad = self._t3_1 * 2 * (self.c2 - np.cos(x2)) * np.sin(x2)
		_t3_2_grad = np.cos(x1)
		grad[:, 0, 0] = 2 * self._t1 + self._t2_2 * _t2_1_grad + self._t3_1 * _t3_2_grad
		grad[:, 0, 1] = -2 * self._t1 + self._t2_1 * _t2_2_grad + self._t3_2 * _t3_1_grad
		return grad

class Bohachevsky(Function):
	"""
	Bohachevsky [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky]
	"""
	def __init__(self, c1=2., c2=0.3, c3=3., c4=0.4, c5=4., c6=0.7, d=2, name="Bohachevsky"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.d = c1, c2, c3, c4, c5, c6, d
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-15., 15.]))

	def __call__(self, x):
		r"""A N-d multimodal function

		.. math::
			f(x)=\sum_{i=1}^{n-1}\left[x_i^2 + c_1x_{i+1}^2 - c_2\cos(c_3\pi x_i) - c_4\cos(c_5\pi x_{i+1}) + c_6\right]

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.


		Default constant values are :math:`c = (2., 0.3, 3., 0.4, 4., 0.7)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x_shr = np.concatenate((x[:, 1:], x[:, :1]), axis=1)
		inner = x ** 2 + self.c1 * x_shr ** 2 - self.c2 * np.cos(self.c3 * np.pi * x) - self.c4 * np.cos(self.c5 * np.pi * x_shr) + self.c6
		return np.sum(inner[:, :-1], axis=1, keepdims=True)

	def grad(self, x):
		# Uses a trick where the first and last column cancel where we need them to.
		x1 = np.concatenate((x[:, :-1], np.zeros((x.shape[0], 1))), axis=1)
		x2 = np.concatenate((np.zeros((x.shape[0], 1)), x[:, 1:]), axis=1)
		return (2 * x1 + self.c2 * self.c3 * np.pi * np.sin(self.c3 * np.pi * x1) + self.c1 * 2 * x2 + self.c4 * self.c5 * np.pi * np.sin(self.c5 * np.pi * x2))[:, np.newaxis, :]

class Branin01(Function):
	"""
	Branin01 [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin01]
	"""
	def __init__(self, c1=1.275, c2=5., c3=6., c4=10., c5=5., c6=4., c7=10., name="Branin01"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5., 10.], [0., 15.]]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		.. math::
			f(x)=\left(-c_1 \frac{x_1^{2}}{\pi^{2}} + c_2 \frac{x_1}{\pi} + x_2 - c_3\right)^{2} + \left(c_4 - \frac{c_5}{c_6 \pi} \right) \cos\left(x_1\right) + c_7

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.


		Default constant values are :math:`c = (1.275, 5., 6., 10., 5., 4., 10.)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		self._t1 = -self.c1 * x1 ** 2 / np.pi ** 2 + self.c2 * x1 / np.pi + x2 - self.c3
		self._t2 = self.c4 - self.c5 / (self.c6 * np.pi)
		return (self._t1 ** 2 + self._t2 * np.cos(x1) + self.c7)[:, np.newaxis]

	def grad(self, x):
		_ = self.__call__(x)
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		x1, x2 = x[:, 0], x[:, 1]
		grad[:, 0, 0] = 2 * self._t1 * (-self.c1 * 2 / np.pi ** 2 * x1 + self.c2 / np.pi) - self._t2 * np.sin(x1)
		grad[:, 0, 1] = 2 * self._t1
		return grad

class Branin02(Function):
	"""
	Branin02 [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin02]
	"""
	def __init__(self, c1=1.275, c2=5., c3=6., c4=10., c5=5., c6=4., c7=1., c8=10., name="Branin02"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8 = c1, c2, c3, c4, c5, c6, c7, c8
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-5., 15.]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		.. math::
			f(x)=\left(- c_1 \frac{x_1^{2}}{\pi^{2}} + c_2 \frac{x_1}{\pi} + x_2 -c_3\right)^{2} + \left(c_4 - \frac{c_5}{c_6 \pi} \right) \cos\left(x_1\right) \cos\left(x_2\right) + \log(x_1^2+x_2^2 +c_7) + c_8

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.


		Default constant values are :math:`c = (1.275, 5., 6., 10., 5., 4., 1., 10.)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		self._t1 = -self.c1 * x1 ** 2 / np.pi ** 2 + self.c2 * x1 / np.pi + x2 - self.c3
		self._t2 = self.c4 - self.c5 / (self.c6 * np.pi)
		self._t3 = np.log(x1 ** 2 + x2 ** 2 + self.c7)
		return (self._t1 ** 2 + self._t2 * np.cos(x1) * np.cos(x2) + self._t3 + self.c8)[:, np.newaxis]

	def grad(self, x):
		_ = self.__call__(x)
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		x1, x2 = x[:, 0], x[:, 1]
		grad[:, 0, 0] = 2 * self._t1 * (-self.c1 * 2 / np.pi ** 2 * x1 + self.c2 / np.pi) - self._t2 * np.cos(x2) * np.sin(x1) + 1 / (x1 ** 2 + x2 ** 2 + self.c7) * 2 * x1
		grad[:, 0, 1] = 2 * self._t1 - self._t2 * np.cos(x1) * np.sin(x2) + 1 / (x1 ** 2 + x2 ** 2 + self.c7) * 2 * x2
		return grad

class Brent(Function):
	"""
	Brent [https://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Brent]
	"""
	def __init__(self, c1=10., c2=10., name="Brent"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		.. math::
			f(x)=(x_1 + 10)^2 + (x_2 + 10)^2 + e^{(-x_1^2-x_2^2)}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.


		Default constant values are :math:`c = (10., 10.)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		return ((x1 + self.c1) ** 2 + (x2 + self.c2) ** 2 + np.exp(-x1 ** 2 - x2 ** 2))[:, np.newaxis]

	def grad(self, x):
		x1, x2 = x[:, 0], x[:, 1]
		return (2 * (x + self.c1) + np.exp(-x1 ** 2 - x2 ** 2)[:, np.newaxis] * -2 * x)[:, np.newaxis, :]

class Bukin02(Function):
    """
    Bukin02 [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin02]
    """
    def __init__(self, c1=100.0, c2=0.01, c3=1.0, c4=0.01, c5=10.0, name="Bukin02"):
        super().__init__()
        self.name = name
        self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
        self.dim = 2
        self.outdim = 1

        self.setDimDom(domain=np.array([[-15., -5.], [-3., 3.]]))

    def __call__(self, x):
        r"""A 2-d multimodal function

        ..math::
            f(x)=c_1 (x_2 - c_2 x_1^2 + c_3) + c_4 (x_1 + c_5)^2


        Default constant values are :math:`c = (100.0, 0.01, 1.0, 0.01, 10.0)

        Args:
            x (np.ndarray): Input array :math:`x` of size `(N,2)`.

        Returns:
            np.ndarray: Output array of size `(N,1)`.
        """
        x1, x2 = x[:, 0], x[:, 1]
        return (self.c1*(x[:, 1]-self.c2*x[:, 0]**2.0+self.c3)+self.c4*(x[:, 0]+self.c5)**2.0).reshape(-1, 1)

    def grad(self, x):
        grad = np.zeros((x.shape[0], self.outdim, self.dim))
        grad[:, 0, 0] = -2.0*self.c2*self.c1*x[:, 0]+2.0*self.c4*(x[:, 0]+self.c5)
        grad[:, 0, 1] = self.c1

        return grad

class Bukin04(Function):
	"""
	Bukin04 [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin04]
	"""
	def __init__(self, c1=100.0, c2=0.01, c3=10.0, name="Bukin04"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-15.0, -5.0], [-3.0, 3.0]]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		..math::
			f(x)=c_1 x_2^{2} + c_2 \abs{x_1 + c_3}


		Default constant values are :math:`c = (100.0, 0.01, 10.0)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1*x[:, 1]**(2.0)+self.c2*np.abs(x[:, 0]+self.c3)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = self.c2*np.sign(x[:, 0]+self.c3)
		grad[:, 0, 1] = 2.0*self.c1*x[:, 1]

		return grad


class Bukin6(Function):
	"""
	Bukin6 [https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin6]
	"""
	def __init__(self, c1=100.0, c2=0.01, c3=0.01, c4=10.0, name="Bukin6"):
		super().__init__()
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-15., -5.], [-3., 3.]]))

	def __call__(self, x):
		r"""A nondifferentiable function which has many local minima

		..math::
			f(x)=c_1\sqrt{\abs{x_2-c_2 x_1^2}}+c_3\abs{x_1+c_4}


		Default constant values are :math:`c = (100.0, 0.01, 0.01, 10.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1*np.sqrt(np.abs(x[:, 1]-self.c2*x[:, 0]**2.0))+self.c3*np.abs(x[:, 0]+self.c4)).reshape(-1, 1)

	def grad(self, x):
		x1 = x[:, 0]
		x2 = x[:, 1]

		grad = np.zeros((x.shape[0], self.outdim, self.dim))

		x1t1 = self.c1 * -np.sign(x2 - self.c2 * x1 ** 2) / np.sqrt(np.abs(x2 - self.c2 * x1 ** 2)) * self.c2 * x1
		x1t2 = self.c3 * np.sign(x1 + self.c4)

		x2t1 = self.c1 * np.sign(x2 - self.c2 * x1 ** 2) / (2 * np.sqrt(np.abs(x2 - self.c2 * x1 ** 2)))

		grad[:, 0, 0] = x1t1 + x1t2
		grad[:, 0, 1] = x2t1

		return grad

class CarromTable(Function):
	"""
	CarromTable [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CarromTable]
	"""
	def __init__(self, c1=1/30, c2=2.0, c3=1.0, name="CarromTable"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10.0, 10.0], [-10.0, 10.0]]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		..math::
			f(x)=- c_1 \exp(c_2 \abs{c_3 - \frac{\sqrt{x_1^{2} + x_2^{2}}}{\pi}}) \cos(x_1)^2 \cos(x_2)^2


		Default constant values are :math:`c = (1/30, 2.0, 1.0)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (-self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*np.cos(x[:, 0])**2.0*np.cos(x[:, 1])**2.0).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*self.c2*np.sign(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi))*(1.0)/(2.0*np.pi*np.sqrt(x[:, 0]**2.0+x[:, 1]**2.0))*2.0*x[:, 0]*np.cos(x[:, 0])**2.0*np.cos(x[:, 1])**2.0+self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 0]**(2.0)+x[:, 1]**(2.0)))/(np.pi)))*2.0*np.cos(x[:, 0])*np.sin(x[:, 0])*np.cos(x[:, 1])**2.0
		grad[:, 0, 1] = self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi)))*self.c2*np.sign(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi))*(1.0)/(2.0*np.pi*np.sqrt(x[:, 1]**2.0+x[:, 0]**2.0))*2.0*x[:, 1]*np.cos(x[:, 1])**2.0*np.cos(x[:, 0])**2.0+self.c1*np.exp(self.c2*np.abs(self.c3-(np.sqrt(x[:, 1]**(2.0)+x[:, 0]**(2.0)))/(np.pi)))*2.0*np.cos(x[:, 1])*np.sin(x[:, 1])*np.cos(x[:, 0])**2.0

		return grad

class Chichinadze(Function):
	"""
	Chichinadze [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Chichinadze]
	"""
	def __init__(self, c1=12.0, c2=8.0, c3=2.5, c4=10.0, c5=0.5, c6=11.0, c7=0.2 * np.sqrt(5), c8=0.5, c9=0.5, name="Chichinadze"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9 = c1, c2, c3, c4, c5, c6, c7, c8, c9
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-30.0, 30.0], [-30.0, 30.0]]))

	def __call__(self, x):
		r"""A 2-d multimodal function

		..math::
			f(x)= x_1^{2} - c_1 x_1 + c_2 \sin(c_3 \pi x_1) + c_4 \cos(c_5 \pi x_1) + c_6 - \frac{c_7}{\exp(c_8 (x_2 -c_9)^{2})}


		Default constant values are :math:`c = (12.0, 8.0, 2.5, 10.0, 0.5, 11.0, 0.2 * \sqrt{5}, 0.5, 0.5)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (x[:, 0]**(2.0)-self.c1*x[:, 0]+self.c2*np.sin(self.c3*np.pi*x[:, 0])+self.c4*np.cos(self.c5*np.pi*x[:, 0])+self.c6-(self.c7)/(np.exp(self.c8*(x[:, 1]-self.c9)**(2.0)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = 2.0*x[:, 0]-self.c1+self.c2*self.c3*np.pi*np.cos(self.c3*np.pi*x[:, 0])-self.c4*self.c5*np.pi*np.sin(self.c5*np.pi*x[:, 0])
		grad[:, 0, 1] = (self.c7)/(np.exp(self.c8*(x[:, 1]-self.c9)**(2.0))**2.0)*np.exp(self.c8*(x[:, 1]-self.c9)**(2.0))*self.c8*2.0*(x[:, 1]-self.c9)

		return grad

class Cigar(Function):
	"""
	Cigar [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Cigar]
	"""
	def __init__(self, c1=10 ** 3, d=4, name="Cigar"):
		super().__init__()
		self.name = name
		self.c1, self.d = c1, d
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-100., 100.]))

	def __call__(self, x):
		r"""A N-d multimodal function

		..math::
			f(x)=x_1^2 + c_1 \sum_{i=2}^{n} x_i^2


		Default constant values are :math:`c = (10 ** 3)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (x[:, 0].reshape(-1, 1)**2+self.c1 * np.sum(x[:, 1:]**2, axis=1, keepdims=True))

	def grad(self, x):
		x_modified = np.concatenate((x[:, 0][:, np.newaxis], self.c1 * x[:, 1:]), axis=1)
		grad = 2 * x_modified

		return grad[:, np.newaxis, :]

class Colville(Function):
	"""
	Colville [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Colville]
	"""
	def __init__(self, c1=1.0, c2=100.0, c3=10.1, c4=1.0, c5=1.0, c6=90.0, c7=10.1, c8=1.0, c9=19.8, c10=1.0, name="Colville"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10
		self.dim = 4
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]))

	def __call__(self, x):
		r"""A 4-d multimodal function

		..math::
			f(x)=(x_1 - c_1)^{2} + c_2 (x_1^{2} - x_2)^{2} + c_3 (x_2 - c_4)^{2} + (x_3 - c_5)^{2} + c_6 (x_3^{2} - x_4)^{2} + c_7 (x_4 - c_8)^{2} + c_9 \frac{x_4 - c_10}{x_2}


		Default constant values are :math:`c = (1.0, 100.0, 10.1, 1.0, 1.0, 90.0, 10.1, 1.0, 19.8, 1.0)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,4)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((x[:, 0]-self.c1)**(2.0)+self.c2*(x[:, 0]**(2.0)-x[:, 1])**(2.0)+self.c3*(x[:, 1]-self.c4)**(2.0)+(x[:, 2]-self.c5)**(2.0)+self.c6*(x[:, 2]**(2.0)-x[:, 3])**(2.0)+self.c7*(x[:, 3]-self.c8)**(2.0)+self.c9*(x[:, 3]-self.c10)/(x[:, 1])).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = 2.0*(x[:, 0]-self.c1)+self.c2*2.0*(x[:, 0]**2.0-x[:, 1])*2.0*x[:, 0]
		grad[:, 0, 1] = -2.0*self.c2*(x[:, 0]**2.0-x[:, 1])+2.0*self.c3*(x[:, 1]-self.c4)-(self.c9*(x[:, 3]-self.c10))/(x[:, 1]**2.0)
		grad[:, 0, 2] = 2.0*(x[:, 2]-self.c5)+2.0*self.c6*(x[:, 2]**2.0-x[:, 3])*2.0*x[:, 2]
		grad[:, 0, 3] = -2.0*self.c6*(x[:, 2]**2.0-x[:, 3])+self.c7*2.0*(x[:, 3]-self.c8)+(self.c9)/(x[:, 1])

		return grad

class CosineMixture(Function):
	"""
	CosineMixture [https://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture]
	"""
	def __init__(self, c1=0.1, c2=5.0, d=2, name="CosineMixture"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-1., 1.]))

	def __call__(self, x):
		r"""A N-d multimodal function

		..math::
			f(x)=-c_1 \sum_{i=1}^n \cos(c_2 \pi x_i) - \sum_{i=1}^n x_i^2


		Default constant values are :math:`c = (0.1, 5.0)
		
		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,d)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (-self.c1*np.sum(np.cos(self.c2*np.pi*x), axis=1)-np.sum(x**2.0, axis=1)).reshape(-1, 1)

	def grad(self, x):
		grad = self.c1*self.c2*np.pi*np.sin(self.c2*np.pi*x)-2*x

		return grad[:, np.newaxis, :]

class CrossInTray(Function):
	"""
	Cross-In-Tray [https://www.sfu.ca/~ssurjano/crossit.html] 
	"""
	def __init__(self, c1=0.0001, c2=100., c3=1., c4=0.1, name="CrossInTray"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-10., 10.]))

	def __call__(self, x):
		r"""Cross-In-Tray function, has lots of local minima

		.. math::
			f(x)=-c_1(\abs{\sin(x_1)\sin(x_2)e^{\abs{c_2-\frac{\sqrt{x_1^2+x_2^2}}{\pi}}}}+c_3)^{c_4}


		Default constant values are :math:`c = (0.0001, 100., 1., 0.1)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		inner1 = np.sin(x1) * np.sin(x2)
		inner2 = np.exp(np.abs(self.c2 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

		return (-self.c1 * (np.abs(inner1 * inner2) + self.c3) ** self.c4)[:, np.newaxis]

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		x1, x2 = x[:, 0], x[:, 1]

		dist = np.sqrt(x1 ** 2 + x2 ** 2)
		inner1x1 = np.cos(x1) * np.sin(x2)
		inner1x2 = np.cos(x2) * np.sin(x1)
		inner2x1 = -np.exp(np.abs(self.c2 - dist / np.pi)) * np.sign(self.c2 - dist / np.pi) * x1 / (np.pi * dist)
		inner2x2 = -np.exp(np.abs(self.c2 - dist / np.pi)) * np.sign(self.c2 - dist / np.pi) * x2 / (np.pi * dist)
		innerx1 = inner1x1 * np.exp(np.abs(self.c2 - dist / np.pi)) + np.sin(x1) * np.sin(x2) * inner2x1
		innerx2 = inner1x2 * np.exp(np.abs(self.c2 - dist / np.pi)) + np.sin(x1) * np.sin(x2) * inner2x2
		inner_absx1 = np.sign(np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi)))) * innerx1
		inner_absx2 = np.sign(np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi)))) * innerx2

		grad[:, 0, 0] = -self.c1 * self.c4 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi))) + self.c3) ** (self.c4 - 1) * inner_absx1
		grad[:, 0, 1] = -self.c1 * self.c4 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(self.c2 - dist / np.pi))) + self.c3) ** (self.c4 - 1) * inner_absx2

		return grad

class Damavandi(Function):
	"""
	Damavandi [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Damavandi]
	"""
	def __init__(self, c1=1.0, c2=2.0, c3=2.0, c4=2.0, c5=2.0, c6=5.0, c7=2.0, c8=7.0, c9=2.0, c10=7.0, name="Damavandi"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[0.0, 14.0], [0.0, 14.0]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left[ c_1 - \abs{\frac{\sin[\pi(x_1-c_2)]\sin[\pi(x_2-c_3)]}{\pi^2(x_1-c_4)(x_2-c_5)}}^{c_6} \right] \left[c_7 + (x_1-c_8)^2 + c_9(x_2-c_{10})^2 \right]


		Default constant values are :math:`c = (1.0, 2.0, 2.0, 2.0, 2.0, 5.0, 2.0, 7.0, 2.0, 7.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((-(((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6))*((self.c6)*((1/(np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))))*((np.sign(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))*((((((np.cos(np.pi*((x[:, 0])-(self.c2))))*(np.pi*1))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))-(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*1)*((x[:, 1])-(self.c5)))))/((((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))**2)))))))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2))))+(((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*((((x[:, 0])-(self.c8))**2)*(2*((1/((x[:, 0])-(self.c8)))*1))))
		grad[:, 0, 1] = ((-(((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6))*((self.c6)*((1/(np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))))*((np.sign(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))*(((((np.sin(np.pi*((x[:, 0])-(self.c2))))*((np.cos(np.pi*((x[:, 1])-(self.c3))))*(np.pi*1)))*(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5))))-(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))*(((np.pi**2)*((x[:, 0])-(self.c4)))*1)))/((((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))**2)))))))*(((self.c7)+(((x[:, 0])-(self.c8))**2))+((self.c9)*(((x[:, 1])-(self.c10))**2))))+(((self.c1)-((np.abs(((np.sin(np.pi*((x[:, 0])-(self.c2))))*(np.sin(np.pi*((x[:, 1])-(self.c3)))))/(((np.pi**2)*((x[:, 0])-(self.c4)))*((x[:, 1])-(self.c5)))))**(self.c6)))*((self.c9)*((((x[:, 1])-(self.c10))**2)*(2*((1/((x[:, 1])-(self.c10)))*1)))))

		return grad

class DeckkersAarts(Function):
	"""
	DeckkersAarts [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeckkersAarts]
	"""
	def __init__(self, c1=1000, c2=0.001, name="DeckkersAarts"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-20.0, 20.0], [-20.0, 20.0]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2 + c_2(x_1^2 + x_2^2)^4


		Default constant values are :math:`c = (1000, 0.001)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((((self.c1)*(x[:, 0]**2))+(x[:, 1]**2))-(((x[:, 0]**2)+(x[:, 1]**2))**2))+((self.c2)*(((x[:, 0]**2)+(x[:, 1]**2))**4))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((self.c1)*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1))))-(((((x[:, 0])**2)+((x[:, 1])**2))**2)*(2*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1)))))))+((self.c2)*(((((x[:, 0])**2)+((x[:, 1])**2))**4)*(4*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 0])**2)*(2*((1/(x[:, 0]))*1)))))))
		grad[:, 0, 1] = ((((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))-(((((x[:, 0])**2)+((x[:, 1])**2))**2)*(2*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))))))+((self.c2)*(((((x[:, 0])**2)+((x[:, 1])**2))**4)*(4*((1/(((x[:, 0])**2)+((x[:, 1])**2)))*(((x[:, 1])**2)*(2*((1/(x[:, 1]))*1)))))))

		return grad

class Dolan(Function):
	"""
	Dolan [https://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Dolan]
	"""
	def __init__(self, c1=1.7, c2=1.5, c3=0.1, c4=0.2, c5=1, name="Dolan"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 5
		self.outdim = 1

		self.setDimDom(domain=np.array([[-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\abs{(x_1 + c_1x_2)\sin(x_1) - c_2x_3 - c_3x_4\cos(x_5 + x_5 - x_1) + c_4x_5^2 - x_2 - c_5}


		Default constant values are :math:`c = (1.7, 1.5, 0.1, 0.2, 1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,5)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.abs(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((np.sin(x[:, 0])+((x[:, 0]+(self.c1*x[:, 1]))*np.cos(x[:, 0])))-((self.c3*x[:, 3])*(-(-np.sin((x[:, 4]+x[:, 4])-x[:, 0])))))
		grad[:, 0, 1] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((self.c1*np.sin(x[:, 0]))-1)
		grad[:, 0, 2] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*(-self.c2)
		grad[:, 0, 3] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*(-(self.c3*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))
		grad[:, 0, 4] = np.sign(((((((x[:, 0]+(self.c1*x[:, 1]))*np.sin(x[:, 0]))-(self.c2*x[:, 2]))-((self.c3*x[:, 3])*np.cos((x[:, 4]+x[:, 4])-x[:, 0])))+(self.c4*x[:, 4]**2))-x[:, 1])-self.c5)*((-((self.c3*x[:, 3])*((-np.sin((x[:, 4]+x[:, 4])-x[:, 0]))*(1+1))))+(self.c4*(x[:, 4]**2*(2*(1/x[:, 4])))))

		return grad

class EggCrate(Function):
	"""
	EggCrate [https://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.EggCrate]
	"""
	def __init__(self, c1=25.0, name="EggCrate"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=x_1^2 + x_2^2 + c_1 \left[ \sin^2(x_1) + \sin^2(x_2) \right]


		Default constant values are :math:`c = (25.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((x[:, 0]**2+x[:, 1]**2)+(self.c1*(np.sin(x[:, 0])**2+np.sin(x[:, 1])**2))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (x[:, 0]**2*(2*(1/x[:, 0])))+(self.c1*(np.sin(x[:, 0])**2*(2*((1/np.sin(x[:, 0]))*np.cos(x[:, 0])))))
		grad[:, 0, 1] = (x[:, 1]**2*(2*(1/x[:, 1])))+(self.c1*(np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1])))))

		return grad

class ElAttarVidyasagarDutta(Function):
	"""
	ElAttarVidyasagarDutta [https://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.ElAttarVidyasagarDutta]
	"""
	def __init__(self, c1=10.0, c2=7.0, c3=1.0, name="ElAttarVidyasagarDutta"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2 + (x_1^2 + x_2^3 - c_3)^2


		Default constant values are :math:`c = (10.0, 7.0, 1.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((x[:, 0]**2+x[:, 1])-self.c1)**2+((x[:, 0]+x[:, 1]**2)-self.c2)**2)+((x[:, 0]**2+x[:, 1]**3)-self.c3)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*((1/((x[:, 0]**2+x[:, 1])-self.c1))*(x[:, 0]**2*(2*(1/x[:, 0]))))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*(1/((x[:, 0]+x[:, 1]**2)-self.c2)))))+(((x[:, 0]**2+x[:, 1]**3)-self.c3)**2*(2*((1/((x[:, 0]**2+x[:, 1]**3)-self.c3))*(x[:, 0]**2*(2*(1/x[:, 0]))))))
		grad[:, 0, 1] = ((((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*(1/((x[:, 0]**2+x[:, 1])-self.c1))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*((1/((x[:, 0]+x[:, 1]**2)-self.c2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))+(((x[:, 0]**2+x[:, 1]**3)-self.c3)**2*(2*((1/((x[:, 0]**2+x[:, 1]**3)-self.c3))*(x[:, 1]**3*(3*(1/x[:, 1]))))))

		return grad

class FreudensteinRoth(Function):
	"""
	FreudensteinRoth [https://infinity77.net/global_optimization/test_functions_nd_F.html#go_benchmark.FreudensteinRoth]
	"""
	def __init__(self, c1=13.0, c2=5.0, c3=2.0, c4=29.0, c5=1.0, c6=14.0, name="FreudensteinRoth"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left(x_1 - c_1 + \left[(c_2 - x_2)x_2 - c_3 \right] x_2 \right)^2 + \left (x_1 - c_4 + \left[(x_2 + c_5)x_2 - c_6 \right] x_2 \right)^2


		Default constant values are :math:`c = (13.0, 5.0, 2.0, 29.0, 1.0, 14.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2+((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2*(2*(1/((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1])))))+(((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2*(2*(1/((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1])))))
		grad[:, 0, 1] = (((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1]))**2*(2*((1/((x[:, 0]-self.c1)+((((self.c2-x[:, 1])*x[:, 1])-self.c3)*x[:, 1])))*((((-x[:, 1])+(self.c2-x[:, 1]))*x[:, 1])+(((self.c2-x[:, 1])*x[:, 1])-self.c3)))))+(((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1]))**2*(2*((1/((x[:, 0]-self.c4)+((((x[:, 1]+self.c5)*x[:, 1])-self.c6)*x[:, 1])))*(((x[:, 1]+(x[:, 1]+self.c5))*x[:, 1])+(((x[:, 1]+self.c5)*x[:, 1])-self.c6)))))

		return grad

class GoldsteinPrice(Function):
	"""
	GoldsteinPrice [https://infinity77.net/global_optimization/test_functions_nd_G.html#go_benchmark.GoldsteinPrice]
	"""
	def __init__(self, name="GoldsteinPrice"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left[ 1+(x_1+x_2+1)^2(19-14x_1+3x_1^2-14x_2+6x_1x_2+3x_2^2) \right] \left[ 30+(2x_1-3x_2)^2(18-32x_1+12x_1^2+48x_2-36x_1x_2+27x_2^2) \right]

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2))))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((((((x[:, 0]+x[:, 1])+1)**2*(2*(1/((x[:, 0]+x[:, 1])+1))))*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2)))+(((x[:, 0]+x[:, 1])+1)**2*(((-14)+(3*(x[:, 0]**2*(2*(1/x[:, 0])))))+(6*x[:, 1]))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))))+((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(((((2*x[:, 0])-(3*x[:, 1]))**2*(2*((1/((2*x[:, 0])-(3*x[:, 1])))*2)))*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))+(((2*x[:, 0])-(3*x[:, 1]))**2*(((-32)+(12*(x[:, 0]**2*(2*(1/x[:, 0])))))-(36*x[:, 1])))))
		grad[:, 0, 1] = ((((((x[:, 0]+x[:, 1])+1)**2*(2*(1/((x[:, 0]+x[:, 1])+1))))*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2)))+(((x[:, 0]+x[:, 1])+1)**2*(((-14)+(6*x[:, 0]))+(3*(x[:, 1]**2*(2*(1/x[:, 1])))))))*(30+(((2*x[:, 0])-(3*x[:, 1]))**2*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))))+((1+(((x[:, 0]+x[:, 1])+1)**2*(((((19-(14*x[:, 0]))+(3*x[:, 0]**2))-(14*x[:, 1]))+((6*x[:, 0])*x[:, 1]))+(3*x[:, 1]**2))))*(((((2*x[:, 0])-(3*x[:, 1]))**2*(2*((1/((2*x[:, 0])-(3*x[:, 1])))*(-3))))*(((((18-(32*x[:, 0]))+(12*x[:, 0]**2))+(48*x[:, 1]))-((36*x[:, 0])*x[:, 1]))+(27*x[:, 1]**2)))+(((2*x[:, 0])-(3*x[:, 1]))**2*((48-(36*x[:, 0]))+(27*(x[:, 1]**2*(2*(1/x[:, 1]))))))))

		return grad

class HimmelBlau(Function):
	"""
	HimmelBlau [https://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.HimmelBlau]
	"""
	def __init__(self, c1=11.0, c2=7.0, name="HimmelBlau"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-6, 6], [-6, 6]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2


		Default constant values are :math:`c = (11.0, 7.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((x[:, 0]**2+x[:, 1])-self.c1)**2+((x[:, 0]+x[:, 1]**2)-self.c2)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*((1/((x[:, 0]**2+x[:, 1])-self.c1))*(x[:, 0]**2*(2*(1/x[:, 0]))))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*(1/((x[:, 0]+x[:, 1]**2)-self.c2))))
		grad[:, 0, 1] = (((x[:, 0]**2+x[:, 1])-self.c1)**2*(2*(1/((x[:, 0]**2+x[:, 1])-self.c1))))+(((x[:, 0]+x[:, 1]**2)-self.c2)**2*(2*((1/((x[:, 0]+x[:, 1]**2)-self.c2))*(x[:, 1]**2*(2*(1/x[:, 1]))))))

		return grad

class Hosaki(Function):
	"""
	Hosaki [https://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.Hosaki]
	"""
	def __init__(self, c1=1.0, c2=8.0, c3=7.0, c4=2.3333333333333335, c5=0.25, name="Hosaki"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[0, 10], [0, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left ( c_1 - c_2x_1 + c_3x_1^2 - c_4x_1^3 + c_5x_1^4 \right )x_2^2e^{-x_1}


		Default constant values are :math:`c = (1.0, 8.0, 7.0, 2.3333333333333335, 0.25)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*x[:, 1]**2)*np.exp(-x[:, 0])).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((((((-self.c2)+(self.c3*(x[:, 0]**2*(2*(1/x[:, 0])))))-(self.c4*(x[:, 0]**3*(3*(1/x[:, 0])))))+(self.c5*(x[:, 0]**4*(4*(1/x[:, 0])))))*x[:, 1]**2)*np.exp(-x[:, 0]))+((((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*x[:, 1]**2)*(-np.exp(-x[:, 0])))
		grad[:, 0, 1] = (((((self.c1-(self.c2*x[:, 0]))+(self.c3*x[:, 0]**2))-(self.c4*x[:, 0]**3))+(self.c5*x[:, 0]**4))*(x[:, 1]**2*(2*(1/x[:, 1]))))*np.exp(-x[:, 0])

		return grad

class Keane(Function):
	"""
	Keane [https://infinity77.net/global_optimization/test_functions_nd_K.html#go_benchmark.Keane]
	"""
	def __init__(self, name="Keane"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[0, 10], [0, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}{\sqrt{x_1^2 + x_2^2}}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((np.sin(x[:, 0]-x[:, 1])**2*np.sin(x[:, 0]+x[:, 1])**2)/np.sqrt(x[:, 0]**2+x[:, 1]**2)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((((np.sin(x[:, 0]-x[:, 1])**2*(2*((1/np.sin(x[:, 0]-x[:, 1]))*np.cos(x[:, 0]-x[:, 1]))))*np.sin(x[:, 0]+x[:, 1])**2)+(np.sin(x[:, 0]-x[:, 1])**2*(np.sin(x[:, 0]+x[:, 1])**2*(2*((1/np.sin(x[:, 0]+x[:, 1]))*np.cos(x[:, 0]+x[:, 1]))))))*np.sqrt(x[:, 0]**2+x[:, 1]**2))-((np.sin(x[:, 0]-x[:, 1])**2*np.sin(x[:, 0]+x[:, 1])**2)*((1/(2*np.sqrt(x[:, 0])))*(x[:, 0]**2*(2*(1/x[:, 0]))))))/np.sqrt(x[:, 0]**2+x[:, 1]**2)**2
		grad[:, 0, 1] = (((((np.sin(x[:, 0]-x[:, 1])**2*(2*((1/np.sin(x[:, 0]-x[:, 1]))*(-np.cos(x[:, 0]-x[:, 1])))))*np.sin(x[:, 0]+x[:, 1])**2)+(np.sin(x[:, 0]-x[:, 1])**2*(np.sin(x[:, 0]+x[:, 1])**2*(2*((1/np.sin(x[:, 0]+x[:, 1]))*np.cos(x[:, 0]+x[:, 1]))))))*np.sqrt(x[:, 0]**2+x[:, 1]**2))-((np.sin(x[:, 0]-x[:, 1])**2*np.sin(x[:, 0]+x[:, 1])**2)*((1/(2*np.sqrt(x[:, 1])))*(x[:, 1]**2*(2*(1/x[:, 1]))))))/np.sqrt(x[:, 0]**2+x[:, 1]**2)**2

		return grad

class Leon(Function):
	"""
	Leon [https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Leon]
	"""
	def __init__(self, c1=100, name="Leon"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-1.2, 1.2], [-1.2, 1.2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)= \left(1 - x_{1}\right)^{2} + c_1 \left(x_{2} - x_{1}^{2} \right)^{2}


		Default constant values are :math:`c = (100)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((1-x[:, 0])**2+(self.c1*(x[:, 1]-x[:, 0]**2)**2)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((1-x[:, 0])**2*(2*(-(1/(1-x[:, 0])))))+(self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))
		grad[:, 0, 1] = self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2))))

		return grad

class Levy13(Function):
	"""
	Levy13 [https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Levy13]
	"""
	def __init__(self, c1=1.0, c2=3.0, c3=1.0, c4=1.0, c5=2.0, c6=1.0, c7=3.0, name="Levy13"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left(x_{1} -c_1\right)^{2} \left[\sin^{2}\left(c_2 \pi x_{2}\right) + c_3\right] + \left(x_{2} -c_4\right)^{2} \left[\sin^{2}\left(c_5 \pi x_{2}\right) + c_6\right] + \sin^{2}\left(c_7 \pi x_{1}\right)


		Default constant values are :math:`c = (1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((x[:, 0]-self.c1)**2*(np.sin((self.c2*np.pi)*x[:, 1])**2+self.c3))+((x[:, 1]-self.c4)**2*(np.sin((self.c5*np.pi)*x[:, 1])**2+self.c6)))+np.sin((self.c7*np.pi)*x[:, 0])**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((x[:, 0]-self.c1)**2*(2*(1/(x[:, 0]-self.c1))))*(np.sin((self.c2*np.pi)*x[:, 1])**2+self.c3))+(np.sin((self.c7*np.pi)*x[:, 0])**2*(2*((1/np.sin((self.c7*np.pi)*x[:, 0]))*(np.cos((self.c7*np.pi)*x[:, 0])*(self.c7*np.pi)))))
		grad[:, 0, 1] = ((x[:, 0]-self.c1)**2*(np.sin((self.c2*np.pi)*x[:, 1])**2*(2*((1/np.sin((self.c2*np.pi)*x[:, 1]))*(np.cos((self.c2*np.pi)*x[:, 1])*(self.c2*np.pi))))))+((((x[:, 1]-self.c4)**2*(2*(1/(x[:, 1]-self.c4))))*(np.sin((self.c5*np.pi)*x[:, 1])**2+self.c6))+((x[:, 1]-self.c4)**2*(np.sin((self.c5*np.pi)*x[:, 1])**2*(2*((1/np.sin((self.c5*np.pi)*x[:, 1]))*(np.cos((self.c5*np.pi)*x[:, 1])*(self.c5*np.pi)))))))

		return grad

class Matyas(Function):
	"""
	Matyas [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Matyas]
	"""
	def __init__(self, c1=0.26, c2=0.48, name="Matyas"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1(x_1^2 + x_2^2) - c_2x_1x_2


		Default constant values are :math:`c = (0.26, 0.48)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((self.c1*(x[:, 0]**2+x[:, 1]**2))-((self.c2*x[:, 0])*x[:, 1])).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*x[:, 1])
		grad[:, 0, 1] = (self.c1*(x[:, 1]**2*(2*(1/x[:, 1]))))-(self.c2*x[:, 0])

		return grad

class McCormick(Function):
	"""
	McCormick [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.McCormick]
	"""
	def __init__(self, c1=2.0, c2=1.0, name="McCormick"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-1.5, 4], [-1.5, 4]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=- x_{1} + c_1 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + c_2


		Default constant values are :math:`c = (2.0, 1.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((((-x[:, 0])+(self.c1*x[:, 1]))+(x[:, 0]-x[:, 1])**2)+np.sin(x[:, 0]+x[:, 1]))+self.c2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((-1)+((x[:, 0]-x[:, 1])**2*(2*(1/(x[:, 0]-x[:, 1])))))+np.cos(x[:, 0]+x[:, 1])
		grad[:, 0, 1] = (self.c1+((x[:, 0]-x[:, 1])**2*(2*(-(1/(x[:, 0]-x[:, 1]))))))+np.cos(x[:, 0]+x[:, 1])

		return grad

class MieleCantrell(Function):
	"""
	MieleCantrell [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.MieleCantrell]
	"""
	def __init__(self, c1=100.0, name="MieleCantrell"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 4
		self.outdim = 1

		self.setDimDom(domain=np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(e^{-x_1} - x_2)^4 + c_1(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8


		Default constant values are :math:`c = (100.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,4)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((np.exp(-x[:, 0])-x[:, 1])**4+(self.c1*(x[:, 1]-x[:, 2])**6))+(np.sin(x[:, 2]-x[:, 3])/np.cos(x[:, 2]-x[:, 3]))**4)+x[:, 0]**8).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((np.exp(-x[:, 0])-x[:, 1])**4*(4*((1/(np.exp(-x[:, 0])-x[:, 1]))*(-np.exp(-x[:, 0])))))+(x[:, 0]**8*(8*(1/x[:, 0])))
		grad[:, 0, 1] = ((np.exp(-x[:, 0])-x[:, 1])**4*(4*(-(1/(np.exp(-x[:, 0])-x[:, 1])))))+(self.c1*((x[:, 1]-x[:, 2])**6*(6*(1/(x[:, 1]-x[:, 2])))))
		grad[:, 0, 2] = (self.c1*((x[:, 1]-x[:, 2])**6*(6*(-(1/(x[:, 1]-x[:, 2]))))))+(np.sin(x[:, 2]-x[:, 3])**4*(4*((1/(np.sin(x[:, 2]-x[:, 3])/np.cos(x[:, 2]-x[:, 3])))*(((np.cos(x[:, 2]-x[:, 3])*np.cos(x[:, 2]-x[:, 3]))-(np.sin(x[:, 2]-x[:, 3])*(-np.sin(x[:, 2]-x[:, 3]))))/np.cos(x[:, 2]-x[:, 3])**2))))
		grad[:, 0, 3] = np.sin(x[:, 2]-x[:, 3])**4*(4*((1/(np.sin(x[:, 2]-x[:, 3])/np.cos(x[:, 2]-x[:, 3])))*((((-np.cos(x[:, 2]-x[:, 3]))*np.cos(x[:, 2]-x[:, 3]))-(np.sin(x[:, 2]-x[:, 3])*(-(-np.sin(x[:, 2]-x[:, 3])))))/np.cos(x[:, 2]-x[:, 3])**2)))

		return grad

class Mishra03(Function):
	"""
	Mishra03 [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra03]
	"""
	def __init__(self, c1=0.01, name="Mishra03"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\sqrt{\abs{\cos{\sqrt{\abs{x_1^2 + x_2^2}}}}} + c_1(x_1 + x_2)


		Default constant values are :math:`c = (0.01)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2)))))+(self.c1*(x[:, 0]+x[:, 1]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((1/(2*np.sqrt(x[:, 0])))*(np.sign(np.cos(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*((-np.sin(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*((1/(2*np.sqrt(x[:, 0])))*(np.sign(x[:, 0]**2+x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))+self.c1
		grad[:, 0, 1] = ((1/(2*np.sqrt(x[:, 1])))*(np.sign(np.cos(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*((-np.sin(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*((1/(2*np.sqrt(x[:, 1])))*(np.sign(x[:, 0]**2+x[:, 1]**2)*(x[:, 1]**2*(2*(1/x[:, 1]))))))))+self.c1

		return grad

class Mishra04(Function):
	"""
	Mishra04 [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra04]
	"""
	def __init__(self, c1=0.01, name="Mishra04"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\sqrt{\abs{\sin{\sqrt{\abs{x_1^2 + x_2^2}}}}} + c_1(x_1 + x_2)


		Default constant values are :math:`c = (0.01)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.sqrt(np.abs(np.sin(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2)))))+(self.c1*(x[:, 0]+x[:, 1]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((1/(2*np.sqrt(x[:, 0])))*(np.sign(np.sin(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*(np.cos(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2)))*((1/(2*np.sqrt(x[:, 0])))*(np.sign(x[:, 0]**2+x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))+self.c1
		grad[:, 0, 1] = ((1/(2*np.sqrt(x[:, 1])))*(np.sign(np.sin(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2))))*(np.cos(np.sqrt(np.abs(x[:, 0]**2+x[:, 1]**2)))*((1/(2*np.sqrt(x[:, 1])))*(np.sign(x[:, 0]**2+x[:, 1]**2)*(x[:, 1]**2*(2*(1/x[:, 1]))))))))+self.c1

		return grad

class Mishra05(Function):
	"""
	Mishra05 [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra05]
	"""
	def __init__(self, c1=0.01, name="Mishra05"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + c_1(x_1 + x_2)


		Default constant values are :math:`c = (0.01)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2+(self.c1*(x[:, 0]+x[:, 1]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0])))))))))+1))))+self.c1
		grad[:, 0, 1] = (((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))+self.c1

		return grad

class Mishra06(Function):
	"""
	Mishra06 [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra06]
	"""
	def __init__(self, c1=0.01, c2=1.0, c3=1.0, name="Mishra06"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=-\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + c_1 \left[(x_1 -c_2)^2 + (x_2 - c_3)^2 \right]


		Default constant values are :math:`c = (0.01, 1.0, 1.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((-np.log(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2))+(self.c1*((x[:, 0]-self.c2)**2+(x[:, 1]-self.c3)**2))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (-((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2)*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0])))))))))-(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0])))))))))+1))))))+(self.c1*((x[:, 0]-self.c2)**2*(2*(1/(x[:, 0]-self.c2)))))
		grad[:, 0, 1] = (-((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2)*(((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0])**2*(2*((1/((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2-np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)+x[:, 0]))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))-(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))))+(self.c1*((x[:, 1]-self.c3)**2*(2*(1/(x[:, 1]-self.c3)))))

		return grad

class Mishra08(Function):
	"""
	Mishra08 [https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra08]
	"""
	def __init__(self, name="Mishra08"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=0.001 \left[\abs{ x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3 + 11520x_1^2 - 5120x_1 + 2624 } \abs{ x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 } \right]^2

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (0).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = 0
		grad[:, 0, 1] = 0

		return grad

class NewFunction03(Function):
	"""
	NewFunction03 [https://infinity77.net/global_optimization/test_functions_nd_N.html#go_benchmark.NewFunction03]
	"""
	def __init__(self, c1=0.01, c2=0.1, name="NewFunction03"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 x_{1} + c_2 x_{2} + \left[x_{1} + \sin^{2}\left[\left(\cos\left(x_{1}\right) + \cos\left(x_{2}\right)\right)^{2}\right] + \cos^{2}\left[\left(\sin\left(x_{1}\right) + \sin\left(x_{2}\right)\right)^{2}\right]\right]^{2}


		Default constant values are :math:`c = (0.01, 0.1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1*x[:, 0])+(self.c2*x[:, 1]))+((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = self.c1+(((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2*(2*((1/((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2))*((1+(np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 0]))))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 0]))))))))))))
		grad[:, 0, 1] = self.c2+(((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2)**2*(2*((1/((x[:, 0]+np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2)+np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2))*((np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)**2*(2*((1/np.sin((np.cos(x[:, 0])+np.cos(x[:, 1]))**2))*(np.cos((np.cos(x[:, 0])+np.cos(x[:, 1]))**2)*((np.cos(x[:, 0])+np.cos(x[:, 1]))**2*(2*((1/(np.cos(x[:, 0])+np.cos(x[:, 1])))*(-np.sin(x[:, 1])))))))))+(np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2)**2*(2*((1/np.cos((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((-np.sin((np.sin(x[:, 0])+np.sin(x[:, 1]))**2))*((np.sin(x[:, 0])+np.sin(x[:, 1]))**2*(2*((1/(np.sin(x[:, 0])+np.sin(x[:, 1])))*np.cos(x[:, 1]))))))))))))

		return grad

class Parsopoulos(Function):
	"""
	Parsopoulos [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Parsopoulos]
	"""
	def __init__(self, name="Parsopoulos"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\cos(x_1)^2 + \sin(x_2)^2

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.cos(x[:, 0])**2+np.sin(x[:, 1])**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = np.cos(x[:, 0])**2*(2*((1/np.cos(x[:, 0]))*(-np.sin(x[:, 0]))))
		grad[:, 0, 1] = np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1])))

		return grad

class Powell(Function):
	"""
	Powell [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Powell]
	"""
	def __init__(self, c1=10.0, c2=5.0, c3=2.0, c4=10.0, name="Powell"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 4
		self.outdim = 1

		self.setDimDom(domain=np.array([[-4, 5], [-4, 5], [-4, 5], [-4, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(x_3+c_1x_1)^2+c_2(x_2-x_4)^2+(x_1-c_3x_2)^4+c_4(x_3-x_4)^4


		Default constant values are :math:`c = (10.0, 5.0, 2.0, 10.0)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,4)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((x[:, 2]+(self.c1*x[:, 0]))**2+(self.c2*(x[:, 1]-x[:, 3])**2))+(x[:, 0]-(self.c3*x[:, 1]))**4)+(self.c4*(x[:, 2]-x[:, 3])**4)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((x[:, 2]+(self.c1*x[:, 0]))**2*(2*((1/(x[:, 2]+(self.c1*x[:, 0])))*self.c1)))+((x[:, 0]-(self.c3*x[:, 1]))**4*(4*(1/(x[:, 0]-(self.c3*x[:, 1])))))
		grad[:, 0, 1] = (self.c2*((x[:, 1]-x[:, 3])**2*(2*(1/(x[:, 1]-x[:, 3])))))+((x[:, 0]-(self.c3*x[:, 1]))**4*(4*((1/(x[:, 0]-(self.c3*x[:, 1])))*(-self.c3))))
		grad[:, 0, 2] = ((x[:, 2]+(self.c1*x[:, 0]))**2*(2*(1/(x[:, 2]+(self.c1*x[:, 0])))))+(self.c4*((x[:, 2]-x[:, 3])**4*(4*(1/(x[:, 2]-x[:, 3])))))
		grad[:, 0, 3] = (self.c2*((x[:, 1]-x[:, 3])**2*(2*(-(1/(x[:, 1]-x[:, 3]))))))+(self.c4*((x[:, 2]-x[:, 3])**4*(4*(-(1/(x[:, 2]-x[:, 3]))))))

		return grad

class Price01(Function):
	"""
	Price01 [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price01]
	"""
	def __init__(self, c1=5, c2=5, name="Price01"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(\abs{ x_1 } - c_1)^2 + (\abs{ x_2 } - c_2)^2


		Default constant values are :math:`c = (5, 5)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((np.abs(x[:, 0])-self.c1)**2+(np.abs(x[:, 1])-self.c2)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (np.abs(x[:, 0])-self.c1)**2*(2*((1/(np.abs(x[:, 0])-self.c1))*np.sign(x[:, 0])))
		grad[:, 0, 1] = (np.abs(x[:, 1])-self.c2)**2*(2*((1/(np.abs(x[:, 1])-self.c2))*np.sign(x[:, 1])))

		return grad

class Price02(Function):
	"""
	Price02 [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price02]
	"""
	def __init__(self, c1=1.0, c2=0.1, name="Price02"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 + \sin^2(x_1) + \sin^2(x_2) - c_2e^{(-x_1^2 - x_2^2)}


		Default constant values are :math:`c = (1.0, 0.1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1+np.sin(x[:, 0])**2)+np.sin(x[:, 1])**2)-(self.c2*np.exp((-x[:, 0]**2)-x[:, 1]**2))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (np.sin(x[:, 0])**2*(2*((1/np.sin(x[:, 0]))*np.cos(x[:, 0]))))-(self.c2*(np.exp((-x[:, 0]**2)-x[:, 1]**2)*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))
		grad[:, 0, 1] = (np.sin(x[:, 1])**2*(2*((1/np.sin(x[:, 1]))*np.cos(x[:, 1]))))-(self.c2*(np.exp((-x[:, 0]**2)-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1]))))))

		return grad

class Price03(Function):
	"""
	Price03 [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price03]
	"""
	def __init__(self, c1=100, c2=6.4, c3=0.5, c4=0.6, name="Price03"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1(x_2 - x_1^2)^2 + \left[c_2(x_2 - c_3)^2 - x_1 - c_4 \right]^2


		Default constant values are :math:`c = (100, 6.4, 0.5, 0.6)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((self.c1*(x[:, 1]-x[:, 0]**2)**2)+(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))+((((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2*(2*(-(1/(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)))))
		grad[:, 0, 1] = (self.c1*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2)))))+((((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4)**2*(2*((1/(((self.c2*(x[:, 1]-self.c3)**2)-x[:, 0])-self.c4))*(self.c2*((x[:, 1]-self.c3)**2*(2*(1/(x[:, 1]-self.c3))))))))

		return grad

class Price04(Function):
	"""
	Price04 [https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Price04]
	"""
	def __init__(self, c1=2, c2=6, name="Price04"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(c_1x_1^3x_2 - x_2^3)^2 + (c_2x_1 - x_2^2 + x_2)^2


		Default constant values are :math:`c = (2, 6)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2+(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2*(2*((1/(((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3))*((self.c1*(x[:, 0]**3*(3*(1/x[:, 0]))))*x[:, 1]))))+((((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2*(2*((1/(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1]))*self.c2)))
		grad[:, 0, 1] = ((((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3)**2*(2*((1/(((self.c1*x[:, 0]**3)*x[:, 1])-x[:, 1]**3))*((self.c1*x[:, 0]**3)-(x[:, 1]**3*(3*(1/x[:, 1])))))))+((((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1])**2*(2*((1/(((self.c2*x[:, 0])-x[:, 1]**2)+x[:, 1]))*((-(x[:, 1]**2*(2*(1/x[:, 1]))))+1))))

		return grad

class Quadratic(Function):
	"""
	Quadratic [https://infinity77.net/global_optimization/test_functions_nd_Q.html#go_benchmark.Quadratic]
	"""
	def __init__(self, name="Quadratic"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=-3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2 + 203.64x_2^2 + 182.25x_1x_2

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((((-3803)-(138*x[:, 0]))-(232*x[:, 1]))+(128*x[:, 0]**2))+(203*x[:, 1]**2))+((182*x[:, 0])*x[:, 1])).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((-138)+(128*(x[:, 0]**2*(2*(1/x[:, 0])))))+(182*x[:, 1])
		grad[:, 0, 1] = ((-232)+(203*(x[:, 1]**2*(2*(1/x[:, 1])))))+(182*x[:, 0])

		return grad

class RosenbrockModified(Function):
	"""
	RosenbrockModified [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RosenbrockModified]
	"""
	def __init__(self, c1=74, c2=100, c3=1, c4=400, c5=0.1, name="RosenbrockModified"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 + c_2(x_2 - x_1^2)^2 + (c_3 - x_1)^2 - c_4 e^{-\frac{(x_1+1)^2 + (x_2 + 1)^2}{c_5}}


		Default constant values are :math:`c = (74, 100, 1, 400, 0.1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1+(self.c2*(x[:, 1]-x[:, 0]**2)**2))+(self.c3-x[:, 0])**2)-(self.c4*np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((self.c2*((x[:, 1]-x[:, 0]**2)**2*(2*((1/(x[:, 1]-x[:, 0]**2))*(-(x[:, 0]**2*(2*(1/x[:, 0]))))))))+((self.c3-x[:, 0])**2*(2*(-(1/(self.c3-x[:, 0]))))))-(self.c4*(np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5))*(-((((x[:, 0]+1)**2*(2*(1/(x[:, 0]+1))))*self.c5)/self.c5**2))))
		grad[:, 0, 1] = (self.c2*((x[:, 1]-x[:, 0]**2)**2*(2*(1/(x[:, 1]-x[:, 0]**2)))))-(self.c4*(np.exp(-(((x[:, 0]+1)**2+(x[:, 1]+1)**2)/self.c5))*(-((((x[:, 1]+1)**2*(2*(1/(x[:, 1]+1))))*self.c5)/self.c5**2))))

		return grad

class RotatedEllipse01(Function):
	"""
	RotatedEllipse01 [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RotatedEllipse01]
	"""
	def __init__(self, c1=7, c2=10.392304845413264, c3=13, name="RotatedEllipse01"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1x_1^2 - c_2 x_1x_2 + c_3x_2^2


		Default constant values are :math:`c = (7, 10.392304845413264, 13)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1*x[:, 0]**2)-((self.c2*x[:, 0])*x[:, 1]))+(self.c3*x[:, 1]**2)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*x[:, 1])
		grad[:, 0, 1] = (-(self.c2*x[:, 0]))+(self.c3*(x[:, 1]**2*(2*(1/x[:, 1]))))

		return grad

class RotatedEllipse02(Function):
	"""
	RotatedEllipse02 [https://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.RotatedEllipse02]
	"""
	def __init__(self, name="RotatedEllipse02"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=x_1^2 - x_1x_2 + x_2^2

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((x[:, 0]**2-(x[:, 0]*x[:, 1]))+x[:, 1]**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (x[:, 0]**2*(2*(1/x[:, 0])))-x[:, 1]
		grad[:, 0, 1] = (-x[:, 0])+(x[:, 1]**2*(2*(1/x[:, 1])))

		return grad

class Schaffer01(Function):
	"""
	Schaffer01 [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer01]
	"""
	def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer01"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 + \frac{\sin^2 (x_1^2 + x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


		Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1+((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2)**2)*(np.sin(x[:, 0]**2+x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2))*(np.cos(x[:, 0]**2+x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
		grad[:, 0, 1] = (((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2)**2)*(np.sin(x[:, 0]**2+x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2+x[:, 1]**2))*(np.cos(x[:, 0]**2+x[:, 1]**2)*(x[:, 1]**2*(2*(1/x[:, 1]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2+x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

		return grad

class Schaffer02(Function):
	"""
	Schaffer02 [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer02]
	"""
	def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer02"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 + \frac{\sin^2 (x_1^2 - x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


		Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1+((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2)**2)*(np.sin(x[:, 0]**2-x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
		grad[:, 0, 1] = (((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2)**2)*(np.sin(x[:, 0]**2-x[:, 1]**2)**2*(2*((1/np.sin(x[:, 0]**2-x[:, 1]**2))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1])))))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.sin(x[:, 0]**2-x[:, 1]**2)**2**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

		return grad

class Schaffer04(Function):
	"""
	Schaffer04 [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schaffer04]
	"""
	def __init__(self, c1=0.5, c2=0.5, c3=1, c4=0.001, name="Schaffer04"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-100, 100], [-100, 100]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 + \frac{\cos^2 \left( \sin(x_1^2 - x_2^2) \right ) - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}


		Default constant values are :math:`c = (0.5, 0.5, 1, 0.001)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1+((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2*(2*((1/np.cos(np.sin(x[:, 0]**2-x[:, 1]**2)))*((-np.sin(np.sin(x[:, 0]**2-x[:, 1]**2)))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(x[:, 0]**2*(2*(1/x[:, 0]))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 0]**2*(2*(1/x[:, 0])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2
		grad[:, 0, 1] = (((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2*(2*((1/np.cos(np.sin(x[:, 0]**2-x[:, 1]**2)))*((-np.sin(np.sin(x[:, 0]**2-x[:, 1]**2)))*(np.cos(x[:, 0]**2-x[:, 1]**2)*(-(x[:, 1]**2*(2*(1/x[:, 1])))))))))*(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2)))-((np.cos(np.sin(x[:, 0]**2-x[:, 1]**2))**2-self.c2)*(self.c4*((x[:, 0]**2+x[:, 1]**2)**2*(2*((1/(x[:, 0]**2+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))))))/(self.c3+(self.c4*(x[:, 0]**2+x[:, 1]**2)**2))**2

		return grad

class SchmidtVetters(Function):
	"""
	SchmidtVetters [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.SchmidtVetters]
	"""
	def __init__(self, c1=1, c2=1, c3=2, c4=2, name="SchmidtVetters"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 3
		self.outdim = 1

		self.setDimDom(domain=np.array([[0, 10], [0, 10], [0, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\frac{c_1}{c_2 + (x_1 - x_2)^2} + \sin \left(\frac{\pi x_2 + x_3}{c_3} \right) + e^{\left(\frac{x_1+x_2}{x_2} - c_4\right)^2}


		Default constant values are :math:`c = (1, 1, 2, 2)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,3)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1/(self.c2+(x[:, 0]-x[:, 1])**2))+np.sin(((np.pi*x[:, 1])+x[:, 2])/self.c3))+np.exp((((x[:, 0]+x[:, 1])/x[:, 1])-self.c4)**2)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((-(self.c1*((x[:, 0]-x[:, 1])**2*(2*(1/(x[:, 0]-x[:, 1]))))))/(self.c2+(x[:, 0]-x[:, 1])**2)**2)+(np.exp((((x[:, 0]+x[:, 1])/x[:, 1])-self.c4)**2)*((((x[:, 0]+x[:, 1])/x[:, 1])-self.c4)**2*(2*((1/(((x[:, 0]+x[:, 1])/x[:, 1])-self.c4))*(x[:, 1]/x[:, 1]**2)))))
		grad[:, 0, 1] = (((-(self.c1*((x[:, 0]-x[:, 1])**2*(2*(-(1/(x[:, 0]-x[:, 1])))))))/(self.c2+(x[:, 0]-x[:, 1])**2)**2)+(np.cos(((np.pi*x[:, 1])+x[:, 2])/self.c3)*((np.pi*self.c3)/self.c3**2)))+(np.exp((((x[:, 0]+x[:, 1])/x[:, 1])-self.c4)**2)*((((x[:, 0]+x[:, 1])/x[:, 1])-self.c4)**2*(2*((1/(((x[:, 0]+x[:, 1])/x[:, 1])-self.c4))*((x[:, 1]-(x[:, 0]+x[:, 1]))/x[:, 1]**2)))))
		grad[:, 0, 2] = np.cos(((np.pi*x[:, 1])+x[:, 2])/self.c3)*(self.c3/self.c3**2)

		return grad

class Schwefel36(Function):
	"""
	Schwefel36 [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Schwefel36]
	"""
	def __init__(self, c1=72, c2=2, c3=2, name="Schwefel36"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[0, 500], [0, 500]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=-x_1x_2(c_1 - c_2x_1 - c_3x_2)


		Default constant values are :math:`c = (72, 2, 2)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((-x[:, 0])*x[:, 1])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((-x[:, 1])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1])))+(((-x[:, 0])*x[:, 1])*(-self.c2))
		grad[:, 0, 1] = ((-x[:, 0])*((self.c1-(self.c2*x[:, 0]))-(self.c3*x[:, 1])))+(((-x[:, 0])*x[:, 1])*(-self.c3))

		return grad

class SixHumpCamel(Function):
	"""
	SixHumpCamel [https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.SixHumpCamel]
	"""
	def __init__(self, c1=4, c2=4, c3=2.1, c4=4, c5=0.3333333333333333, name="SixHumpCamel"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1x_1^2+x_1x_2-c_2x_2^2-c_3x_1^4+c_4x_2^4+c_5x_1^6


		Default constant values are :math:`c = (4, 4, 2.1, 4, 0.3333333333333333)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((((self.c1*x[:, 0]**2)+(x[:, 0]*x[:, 1]))-(self.c2*x[:, 1]**2))-(self.c3*x[:, 0]**4))+(self.c4*x[:, 1]**4))+(self.c5*x[:, 0]**6)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))+x[:, 1])-(self.c3*(x[:, 0]**4*(4*(1/x[:, 0])))))+(self.c5*(x[:, 0]**6*(6*(1/x[:, 0]))))
		grad[:, 0, 1] = (x[:, 0]-(self.c2*(x[:, 1]**2*(2*(1/x[:, 1])))))+(self.c4*(x[:, 1]**4*(4*(1/x[:, 1]))))

		return grad

class ThreeHumpCamel(Function):
	"""
	ThreeHumpCamel [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.ThreeHumpCamel]
	"""
	def __init__(self, c1=2, c2=1.05, c3=6, name="ThreeHumpCamel"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1x_1^2 - c_2x_1^4 + \frac{x_1^6}{c_3} + x_1x_2 + x_2^2


		Default constant values are :math:`c = (2, 1.05, 6)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((((self.c1*x[:, 0]**2)-(self.c2*x[:, 0]**4))+(x[:, 0]**6/self.c3))+(x[:, 0]*x[:, 1]))+x[:, 1]**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((self.c1*(x[:, 0]**2*(2*(1/x[:, 0]))))-(self.c2*(x[:, 0]**4*(4*(1/x[:, 0])))))+(((x[:, 0]**6*(6*(1/x[:, 0])))*self.c3)/self.c3**2))+x[:, 1]
		grad[:, 0, 1] = x[:, 0]+(x[:, 1]**2*(2*(1/x[:, 1])))

		return grad

class Treccani(Function):
	"""
	Treccani [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.Treccani]
	"""
	def __init__(self, c1=4, c2=4, name="Treccani"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=x_1^4 + c_1x_1^3 + c_2x_1^2 + x_2^2


		Default constant values are :math:`c = (4, 4)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((x[:, 0]**4+(self.c1*x[:, 0]**3))+(self.c2*x[:, 0]**2))+x[:, 1]**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((x[:, 0]**4*(4*(1/x[:, 0])))+(self.c1*(x[:, 0]**3*(3*(1/x[:, 0])))))+(self.c2*(x[:, 0]**2*(2*(1/x[:, 0]))))
		grad[:, 0, 1] = x[:, 1]**2*(2*(1/x[:, 1]))

		return grad

class Trefethen(Function):
	"""
	Trefethen [https://infinity77.net/global_optimization/test_functions_nd_T.html#go_benchmark.Trefethen]
	"""
	def __init__(self, name="Trefethen"):
		super().__init__()
		self.name = name
		
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=0.25 x_{1}^{2} + 0.25 x_{2}^{2} + e^{\sin\left(50 x_{1}\right)} - \sin\left(10 x_{1} + 10 x_{2}\right) + \sin\left(60 e^{x_{2}}\right) + \sin\left[70 \sin\left(x_{1}\right)\right] + \sin\left[\sin\left(80 x_{2}\right)\right]

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((np.exp(np.sin(50*x[:, 0]))-np.sin((10*x[:, 0])+(10*x[:, 1])))+np.sin(60*np.exp(x[:, 1])))+np.sin(70*np.sin(x[:, 0])))+np.sin(np.sin(80*x[:, 1]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((np.exp(np.sin(50*x[:, 0]))*(np.cos(50*x[:, 0])*50))-(np.cos((10*x[:, 0])+(10*x[:, 1]))*10))+(np.cos(70*np.sin(x[:, 0]))*(70*np.cos(x[:, 0])))
		grad[:, 0, 1] = ((-(np.cos((10*x[:, 0])+(10*x[:, 1]))*10))+(np.cos(60*np.exp(x[:, 1]))*(60*np.exp(x[:, 1]))))+(np.cos(np.sin(80*x[:, 1]))*(np.cos(80*x[:, 1])*80))

		return grad

class Ursem01(Function):
	"""
	Ursem01 [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem01]
	"""
	def __init__(self, c1=2, c2=0.5, c3=3, c4=0.5, name="Ursem01"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-2.5, 3], [-2, 2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=- \sin(c_1x_1 - c_2 \pi) - c_3 \cos(x_2) - c_4x_1


		Default constant values are :math:`c = (2, 0.5, 3, 0.5)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((-np.sin((self.c1*x[:, 0])-(self.c2*np.pi)))-(self.c3*np.cos(x[:, 1])))-(self.c4*x[:, 0])).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (-(np.cos((self.c1*x[:, 0])-(self.c2*np.pi))*self.c1))-self.c4
		grad[:, 0, 1] = -(self.c3*(-np.sin(x[:, 1])))

		return grad

class Ursem03(Function):
	"""
	Ursem03 [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem03]
	"""
	def __init__(self, c1=2.2, c2=0.5, c3=2, c4=2, c5=3, c6=2, c7=2.2, c8=0.5, c9=2, c10=2, c11=3, c12=2, name="Ursem03"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10, self.c11, self.c12 = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-2, 2], [-1.5, 1.5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=- \sin(c_1 \pi x_1 + c_2 \pi) \frac{c_3 - \abs{ x_1 }}{c_4} \frac{c_5 - \abs{ x_1 }}{c_6} - \sin(c_7 \pi x_2 + c_8 \pi) \frac{c_9 - \abs{ x_2 }}{c_{10}} \frac{c_{11} - \abs{ x_2 }}{c_{12}}


		Default constant values are :math:`c = (2.2, 0.5, 2, 2, 3, 2, 2.2, 0.5, 2, 2, 3, 2)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))*((self.c5-np.abs(x[:, 0]))/self.c6))-((np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))*((self.c11-np.abs(x[:, 1]))/self.c12))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((((-(np.cos(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi))*(self.c1*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))+((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*(((-np.sign(x[:, 0]))*self.c4)/self.c4**2)))*((self.c5-np.abs(x[:, 0]))/self.c6))+(((-np.sin(((self.c1*np.pi)*x[:, 0])+(self.c2*np.pi)))*((self.c3-np.abs(x[:, 0]))/self.c4))*(((-np.sign(x[:, 0]))*self.c6)/self.c6**2))
		grad[:, 0, 1] = -(((((np.cos(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*(self.c7*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))+(np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*(((-np.sign(x[:, 1]))*self.c10)/self.c10**2)))*((self.c11-np.abs(x[:, 1]))/self.c12))+((np.sin(((self.c7*np.pi)*x[:, 1])+(self.c8*np.pi))*((self.c9-np.abs(x[:, 1]))/self.c10))*(((-np.sign(x[:, 1]))*self.c12)/self.c12**2)))

		return grad

class Ursem04(Function):
	"""
	Ursem04 [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.Ursem04]
	"""
	def __init__(self, c1=3, c2=0.5, c3=0.5, c4=2, c5=4, name="Ursem04"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5 = c1, c2, c3, c4, c5
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-2, 2], [-2, 2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=-c_1 \sin(c_2 \pi x_1 + c_3 \pi) \frac{c_4 - \sqrt{x_1^2 + x_2 ^ 2}}{c_5}


		Default constant values are :math:`c = (3, 0.5, 0.5, 2, 4)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((-self.c1)*np.sin(((self.c2*np.pi)*x[:, 0])+(self.c3*np.pi)))*((self.c4-np.sqrt(x[:, 0]**2+x[:, 1]**2))/self.c5)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((-self.c1)*(np.cos(((self.c2*np.pi)*x[:, 0])+(self.c3*np.pi))*(self.c2*np.pi)))*((self.c4-np.sqrt(x[:, 0]**2+x[:, 1]**2))/self.c5))+(((-self.c1)*np.sin(((self.c2*np.pi)*x[:, 0])+(self.c3*np.pi)))*(((-((1/(2*np.sqrt(x[:, 0])))*(x[:, 0]**2*(2*(1/x[:, 0])))))*self.c5)/self.c5**2))
		grad[:, 0, 1] = ((-self.c1)*np.sin(((self.c2*np.pi)*x[:, 0])+(self.c3*np.pi)))*(((-((1/(2*np.sqrt(x[:, 1])))*(x[:, 1]**2*(2*(1/x[:, 1])))))*self.c5)/self.c5**2)

		return grad

class UrsemWaves(Function):
	"""
	UrsemWaves [https://infinity77.net/global_optimization/test_functions_nd_U.html#go_benchmark.UrsemWaves]
	"""
	def __init__(self, c1=0.9, c2=4.5, c3=4.7, c4=2, c5=2, c6=2.5, name="UrsemWaves"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-0.9, 1.2], [-1.2, 1.2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=-c_1x_1^2 + (x_2^2 - c_2x_2^2)x_1x_2 + c_3 \cos \left[ c_4x_1 - x_2^2(c_5 + x_1) \right ] \sin(c_6 \pi x_1)


		Default constant values are :math:`c = (0.9, 4.5, 4.7, 2, 2, 2.5)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((-self.c1)*x[:, 0]**2)+(((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 0])*x[:, 1]))+((self.c3*np.cos((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*np.sin((self.c6*np.pi)*x[:, 0]))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((-self.c1)*(x[:, 0]**2*(2*(1/x[:, 0]))))+((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 1]))+(((self.c3*((-np.sin((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(self.c4-x[:, 1]**2)))*np.sin((self.c6*np.pi)*x[:, 0]))+((self.c3*np.cos((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(np.cos((self.c6*np.pi)*x[:, 0])*(self.c6*np.pi))))
		grad[:, 0, 1] = (((((x[:, 1]**2*(2*(1/x[:, 1])))-(self.c2*(x[:, 1]**2*(2*(1/x[:, 1])))))*x[:, 0])*x[:, 1])+((x[:, 1]**2-(self.c2*x[:, 1]**2))*x[:, 0]))+((self.c3*((-np.sin((self.c4*x[:, 0])-(x[:, 1]**2*(self.c5+x[:, 0]))))*(-((x[:, 1]**2*(2*(1/x[:, 1])))*(self.c5+x[:, 0])))))*np.sin((self.c6*np.pi)*x[:, 0]))

		return grad

class VenterSobiezcczanskiSobieski(Function):
	"""
	VenterSobiezcczanskiSobieski [https://infinity77.net/global_optimization/test_functions_nd_V.html#go_benchmark.VenterSobiezcczanskiSobieski]
	"""
	def __init__(self, c1=100, c2=100, c3=30, c4=100, c5=100, c6=30, name="VenterSobiezcczanskiSobieski"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-50, 50], [-50, 50]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=x_1^2 - c_1 \cos^2(x_1) - c_2 \cos(x_1^2/c_3) + x_2^2 - c_4 \cos^2(x_2) - c_5 \cos(x_2^2/c_6)


		Default constant values are :math:`c = (100, 100, 30, 100, 100, 30)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((((x[:, 0]**2-(self.c1*np.cos(x[:, 0])**2))-(self.c2*np.cos(x[:, 0]**2/self.c3)))+x[:, 1]**2)-(self.c4*np.cos(x[:, 1])**2))-(self.c5*np.cos(x[:, 1]**2/self.c6))).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((x[:, 0]**2*(2*(1/x[:, 0])))-(self.c1*(np.cos(x[:, 0])**2*(2*((1/np.cos(x[:, 0]))*(-np.sin(x[:, 0])))))))-(self.c2*((-np.sin(x[:, 0]**2/self.c3))*(((x[:, 0]**2*(2*(1/x[:, 0])))*self.c3)/self.c3**2)))
		grad[:, 0, 1] = ((x[:, 1]**2*(2*(1/x[:, 1])))-(self.c4*(np.cos(x[:, 1])**2*(2*((1/np.cos(x[:, 1]))*(-np.sin(x[:, 1])))))))-(self.c5*((-np.sin(x[:, 1]**2/self.c6))*(((x[:, 1]**2*(2*(1/x[:, 1])))*self.c6)/self.c6**2)))

		return grad

class WayburnSeader01(Function):
	"""
	WayburnSeader01 [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.WayburnSeader01]
	"""
	def __init__(self, c1=17, c2=2, c3=4, name="WayburnSeader01"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-5, 5], [-5, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=(x_1^6 + x_2^4 - c_1)^2 + (c_2x_1 + x_2 - c_3)^2


		Default constant values are :math:`c = (17, 2, 4)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2+(((self.c2*x[:, 0])+x[:, 1])-self.c3)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2*(2*((1/((x[:, 0]**6+x[:, 1]**4)-self.c1))*(x[:, 0]**6*(6*(1/x[:, 0]))))))+((((self.c2*x[:, 0])+x[:, 1])-self.c3)**2*(2*((1/(((self.c2*x[:, 0])+x[:, 1])-self.c3))*self.c2)))
		grad[:, 0, 1] = (((x[:, 0]**6+x[:, 1]**4)-self.c1)**2*(2*((1/((x[:, 0]**6+x[:, 1]**4)-self.c1))*(x[:, 1]**4*(4*(1/x[:, 1]))))))+((((self.c2*x[:, 0])+x[:, 1])-self.c3)**2*(2*(1/(((self.c2*x[:, 0])+x[:, 1])-self.c3))))

		return grad

class WayburnSeader02(Function):
	"""
	WayburnSeader02 [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.WayburnSeader02]
	"""
	def __init__(self, c1=1.613, c2=4, c3=0.3125, c4=4, c5=1.625, c6=1, name="WayburnSeader02"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-500, 500], [-500, 500]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=\left[ c_1 - c_2(x_1 - c_3)^2 - c_4(x_2 - c_5)^2 \right]^2 + (x_2 - c_6)^2


		Default constant values are :math:`c = (1.613, 4, 0.3125, 4, 1.625, 1)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2+(x[:, 1]-self.c6)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2*(2*((1/((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2)))*(-(self.c2*((x[:, 0]-self.c3)**2*(2*(1/(x[:, 0]-self.c3))))))))
		grad[:, 0, 1] = (((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2))**2*(2*((1/((self.c1-(self.c2*(x[:, 0]-self.c3)**2))-(self.c4*(x[:, 1]-self.c5)**2)))*(-(self.c4*((x[:, 1]-self.c5)**2*(2*(1/(x[:, 1]-self.c5)))))))))+((x[:, 1]-self.c6)**2*(2*(1/(x[:, 1]-self.c6))))

		return grad

class Wolfe(Function):
	"""
	Wolfe [https://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Wolfe]
	"""
	def __init__(self, c1=1.3333333333333333, c2=0.75, name="Wolfe"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 3
		self.outdim = 1

		self.setDimDom(domain=np.array([[0, 2], [0, 2], [0, 2]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1(x_1^2 + x_2^2 - x_1x_2)^{c_2} + x_3


		Default constant values are :math:`c = (1.3333333333333333, 0.75)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,3)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((self.c1*((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2)+x[:, 2]).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = self.c1*(((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2*(self.c2*((1/((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1])))*((x[:, 0]**2*(2*(1/x[:, 0])))-x[:, 1]))))
		grad[:, 0, 1] = self.c1*(((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1]))**self.c2*(self.c2*((1/((x[:, 0]**2+x[:, 1]**2)-(x[:, 0]*x[:, 1])))*((x[:, 1]**2*(2*(1/x[:, 1])))-x[:, 0]))))
		grad[:, 0, 2] = 1

		return grad

class Zettl(Function):
	"""
	Zettl [https://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.Zettl]
	"""
	def __init__(self, c1=0.25, c2=2, name="Zettl"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-1, 5], [-1, 5]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1 x_{1} + \left(x_{1}^{2} - c_2 x_{1} + x_{2}^{2}\right)^{2}


		Default constant values are :math:`c = (0.25, 2)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((self.c1*x[:, 0])+((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = self.c1+(((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2*(2*((1/((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2))*((x[:, 0]**2*(2*(1/x[:, 0])))-self.c2))))
		grad[:, 0, 1] = ((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2)**2*(2*((1/((x[:, 0]**2-(self.c2*x[:, 0]))+x[:, 1]**2))*(x[:, 1]**2*(2*(1/x[:, 1])))))

		return grad

class Zirilli(Function):
	"""
	Zirilli [https://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.Zirilli]
	"""
	def __init__(self, c1=0.25, c2=0.5, c3=0.1, c4=0.5, name="Zirilli"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.array([[-10, 10], [-10, 10]]))

	def __call__(self, x):
		r"""A multimodal minimzation function

		..math::
			f(x)=c_1x_1^4 - c_2x_1^2 + c_3x_1 + c_4x_2^2


		Default constant values are :math:`c = (0.25, 0.5, 0.1, 0.5)

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return ((((self.c1*x[:, 0]**4)-(self.c2*x[:, 0]**2))+(self.c3*x[:, 0]))+(self.c4*x[:, 1]**2)).reshape(-1, 1)

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		grad[:, 0, 0] = ((self.c1*(x[:, 0]**4*(4*(1/x[:, 0]))))-(self.c2*(x[:, 0]**2*(2*(1/x[:, 0])))))+self.c3
		grad[:, 0, 1] = self.c4*(x[:, 1]**2*(2*(1/x[:, 1])))

		return grad

# https://www.sfu.ca/~ssurjano/optimization.html, many local minima section,
# excluding discontinuous functions

class DropWave(Function):
	"""
	Drop-Wave [https://www.sfu.ca/~ssurjano/drop.html]
	"""
	def __init__(self, c1=1., c2=12., c3=0.5, c4=2., name="DropWave"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-5.12, 5.12]))

	def __call__(self, x):
		r"""Drop Wave function, has a lot of intricate details which make finding the global minimum very difficult

		.. math::
			f(x)=-\frac{c_1+\cos(c_2\sqrt{x_1^2+x_2^2})}{c_3(x_1^2+x_2^2)+c_4}


		Default constant values are :math:`c = (1., 12., 0.5, 2.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._numerator = (self.c1 + np.cos(self.c2 * np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)))[:, np.newaxis]
		self._denominator = (self.c3 * (x[:, 0] ** 2 + x[:, 1] ** 2) + self.c4)[:, np.newaxis]
		return -self._numerator / self._denominator

	def grad(self, x):
		_ = self.__call__(x)
		x1, x2 = x[:, 0], x[:, 1]
		dist_sq = (x1 ** 2 + x2 ** 2)[:, np.newaxis]
		num_grad = -np.sin(self.c2 * np.sqrt(dist_sq)) * self.c2 * x / np.sqrt(dist_sq)
		denom_grad = 2 * self.c3 * x

		return (-(num_grad * self._denominator - self._numerator * denom_grad) / (self._denominator ** 2))[:, np.newaxis, :]

class EggHolder(Function):
	"""
	Egg Holder [https://www.sfu.ca/~ssurjano/egg.html]
	"""
	def __init__(self, c1=47., c2=47., c3=47., name='EggHolder'):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-512., 512.]))

	def __call__(self, x):
		r"""Egg Holder function, has a much larger scale than usual

		.. math::
			f(x)=-(x_2+c_1)\sin(\sqrt{\abs{x_2+\frac{x_1}{2}+c_2}})-x_1\sin(\sqrt{\abs{x_1-(x_2+c_3)}})


		Default constant values are :math:`c = (47., 47., 47.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2 = x[:, 0], x[:, 1]
		self._term1_1 = x2 + self.c1
		self._term1_2 = np.sin(np.sqrt(np.abs(x2 + x1 / 2 + self.c2)))
		self._term2 = x1 * np.sin(x1 - (x2 + self.c3))
		return (-self._term1_1 * self._term1_2 - self._term2)[:, np.newaxis]

	def grad(self, x):
		_ = self.__call__(x)
		grad = np.zeros((x.shape[0], self.outdim, self.dim))

		x1, x2 = x[:, 0], x[:, 1]
		x1_grad1 = self._term1_1 * np.cos(np.sqrt(np.abs(x2 + x1 / 2 + self.c2))) * np.sign(x2 + x1 / 2 + self.c2) / (4 * np.sqrt(np.abs(x2 + x1 / 2 + self.c2)))
		x1_grad2 = np.sin(x1 - (x2 + self.c3)) + x1 * np.cos(x1 - (x2 + self.c3))
		x2_grad1 = x1_grad1 * 2 + self._term1_2
		x2_grad2 = -x1 * np.cos(x1 - x2 - self.c3)
		grad[:, 0, 0] = -x1_grad1 - x1_grad2
		grad[:, 0, 1] = -x2_grad1 - x2_grad2

		return grad

class Griewank(Function):
	"""
	Griewank [https://www.sfu.ca/~ssurjano/griewank.html]
	"""
	def __init__(self, c1=4000., c2=1., d=2, name="Griewank"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.d = c1, c2, d
		self.dim = d
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([-600., 600.]))

	def __call__(self, x):
		r"""Griewank function, is large scale but has very frequent local minima

		.. math::
			f(x)=\sum_{i=1}^{d}\frac{x_i^2}{c_1}-\prod_{i=1}^{d}\cos(\frac{x_i}{\sqrt{i}}) + c_2


		Default constant values are :math:`c = (4000., 1.)` and :math:`d = 2`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,dim)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		self._term1 = np.sum(x ** 2 / self.c1, axis=1, keepdims=True)
		self._term2 = 1.
		for i in range(1, self.dim + 1):
			self._term2 *= np.cos(x[:, i - 1] / np.sqrt(i))
		self._term2 = self._term2

		return self._term1 - self._term2[:, np.newaxis] + self.c2

	def grad(self, x):
		_ = self.__call__(x)
		term1_grad = 2 * x / self.c1
		term2_grad = np.zeros((x.shape[0], self.dim))
		for i in range(1, self.dim + 1):
			term2_grad[:, i - 1] = -self._term2 / np.cos(x[:, i - 1] / np.sqrt(i)) * np.sin(x[:, i - 1] / np.sqrt(i)) / np.sqrt(i)

		return (term1_grad - term2_grad)[:, np.newaxis, :]

# https://www.sfu.ca/~ssurjano/emulat.html, trig section,
# excluding #18 Higdon (2002) and Gramacy & Lee (2008), which is discontinuous

class ChengSandu(Function):
	"""
	Cheng and Sandu [https://www.sfu.ca/~ssurjano/chsan10.html]
	"""
	def __init__(self, name="ChengSandu"):
		super().__init__()
		self.name = name
		self.dim = 2
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""Cheng and Sandu 2d function

		.. math::
			f(x)=\cos(x_1+x_2)e^{x_1x_2}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (np.cos(x[:, 0] + x[:, 1]) * np.exp(x[:, 0] * x[:, 1]))[:, np.newaxis]

	def grad(self, x):
		exp = np.exp(x[:, 0] * x[:, 1])[:, np.newaxis]
		sin = np.sin(x[:, 0] + x[:, 1])[:, np.newaxis]
		cos = np.cos(x[:, 0] + x[:, 1])[:, np.newaxis]

		return ((x[:, ::-1] * cos - sin) * exp)[:, np.newaxis, :]

class Sine1d(Function):
	"""
	Sinusoidal [https://www.sfu.ca/~ssurjano/curretal88sin.html]
	"""
	def __init__(self, c1=2., c2=0.1, name="Sinusoidal"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""Simple 1d sine function


		Default constant values are :math:`c = (2., 0.1)`.

		.. math::
			f(x)=\sin(c_1\pi(x-c_2))

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""

		return np.sin(self.c1 * np.pi * (x - self.c2))

	def grad(self, x):
		return (np.cos(self.c1 * np.pi * (x - self.c2)) * self.c1 * np.pi)[:, np.newaxis, :]

class Forrester(Function):
	"""
	Forrester [https://www.sfu.ca/~ssurjano/forretal08.html]
	"""
	def __init__(self, c1=6., c2=2., c3=12., c4=4., name="Forrester"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 1
		self.outdim = 1
		
		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""Simple 1d function, with several local minima.


		Default constant values are :math:`c = (6., 2., 12., 4)

		.. math::
			f(x)=(c_1x-c_2)^2\sin(c_3x-c_4)


		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1 * x - self.c2) ** 2 * np.sin(self.c3 * x - self.c4)

	def grad(self, x):
		return (2 * self.c1 * (self.c1 * x - self.c2) * np.sin(self.c3 * x - self.c4) + (self.c1 * x - self.c2) ** 2 * self.c3 * np.cos(self.c3 * x - self.c4))[:, np.newaxis, :]

class Friedman(Function):
	"""
	Friedman [https://www.sfu.ca/~ssurjano/fried.html]
	"""
	def __init__(self, c1=10., c2=20., c3=-0.5, c4=2., c5=10., c6=5., name="Friedman"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = c1, c2, c3, c4, c5, c6
		self.dim = 5
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""A 5d trigonometric function, with :math:`x_4` and :math:`x_5` being linear


		Default constant values are :math:`c = (10., 20., 0.5, 2., 10., 5.)

		.. math::
			f(x)=c_1\sin(\pi x_1x_2)+c_2(x_3-c_3)^{c_4}+c_5x_4+c_6x_5

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,5)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
		return (self.c1 * np.sin(np.pi * x1 * x2) + self.c2 * (x3 - self.c3) ** self.c4 + self.c5 * x4 + self.c6 * x5)[:, np.newaxis]

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))

		grad[:, 0, 0] = self.c1 * np.pi * x[:, 1] * np.cos(np.pi * x[:, 0] * x[:, 1])
		grad[:, 0, 1] = self.c1 * np.pi * x[:, 0] * np.cos(np.pi * x[:, 0] * x[:, 1])
		grad[:, 0, 2] = self.c2 * self.c4 * (x[:, 2] - self.c3) ** (self.c4 - 1)
		grad[:, 0, 3] = self.c5
		grad[:, 0, 4] = self.c6

		return grad

class GramacyLee(Function):
	"""
	Gramacy and Lee (2009) [https://www.sfu.ca/~ssurjano/grlee09.html]
	"""
	def __init__(self, c1=0.9, c2=0.48, c3=10., name="GramacyLee"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 6
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""A 6d function, where x5 and x6 aren't active
		
		.. math::
			f(x)=e^{sin((c_1(x_1+c_2))^{c_3})}+x_2x_3+x_4


		Default constant values are :math:`c = (0.9, 0.48, 10.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,6)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
		return (np.exp(np.sin((self.c1 * (x1 + self.c2)) ** self.c3)) + x2 * x3 + x4)[:, np.newaxis]

	def grad(self, x):
		x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
		grad = np.zeros((x.shape[0], self.outdim, self.dim))

		grad[:, 0, 0] = (np.exp(np.sin((self.c1 * (x1 + self.c2)) ** self.c3))) \
						* np.cos((self.c1 * (x1 + self.c2)) ** self.c3) \
						* self.c3 * (self.c1 * (x1 + self.c2)) ** (self.c3 - 1) \
						* self.c1
		grad[:, 0, 1] = x3
		grad[:, 0, 2] = x2
		grad[:, 0, 3] = 1.
		grad[:, 0, 4:] = 0.

		return grad

class GramacyLee2(Function):
	"""
	Gramacy and Lee (2012) [https://www.sfu.ca/~ssurjano/grlee12.html]
	"""
	def __init__(self, c1=10., c2=2., c3=1., c4=4., name="GramacyLee2"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0.5, 2.5]))

	def __call__(self, x):
		r"""Complicated sine-like 1d function
		

		Default constant values are :math:`c = (10., 2., 1., 4.)`.

		.. math::
			f(x)=\frac{\sin(c_1\pi x)}{c_2x}+(x-c_3)^{c_4}

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.sin(self.c1 * np.pi * x) / (self.c2 * x) + (x - self.c3) ** self.c4

	def grad(self, x):
		return (((self.c1 * np.pi * np.cos(self.c1 * np.pi * x) * self.c2 * x) - self.c2 * np.sin(self.c1 * np.pi * x)) / (self.c2 * x) ** 2 + self.c4 * (x - self.c3) ** (self.c4 - 1))[:, np.newaxis, :]

class Higdon(Function):
	"""
	Higdon (2002) [https://www.sfu.ca/~ssurjano/hig02.html]
	"""
	def __init__(self, c1=10., c2=0.2, c3=2.5, name="Higdon"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3 = c1, c2, c3
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

	def __call__(self, x):
		r"""A 1d wavy function
		
		.. math::
			f(x)=\sin(2\pi x/c_1) + c_2\sin(2\pi x/c_3)


		Default constant values are :math:`c = (10., 0.2, 2.5)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.sin(2 * np.pi * x / self.c1) + self.c2 * np.sin(2 * np.pi * x / self.c3)

	def grad(self, x):
		return (2 * np.pi / self.c1 * np.cos(2 * np.pi * x / self.c1) + 2 * np.pi * self.c2 / self.c3 * np.cos(2 * np.pi * x / self.c3))[:, np.newaxis, :]

class Holsclaw(Function):
	"""
	Holsclaw et al. [https://www.sfu.ca/~ssurjano/holsetal13sin.html]
	"""
	def __init__(self, c1=10., name="Holsclaw"):
		super().__init__()
		self.name = name
		self.c1 = c1
		self.dim = 1
		self.outdim = 1


		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 10.]))

	def __call__(self, x):
		r"""A 1d wavy function

		.. math::
			f(x)=\frac{x\sin(x)}{c_1}


		Default constant values are :math:`c = (10.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return x * np.sin(x) / self.c1

	def grad(self, x):
		return ((np.sin(x) + x * np.cos(x)) / self.c1)[:, np.newaxis, :]

class Lim(Function):
	"""
	Lim et al. (2002) [https://www.sfu.ca/~ssurjano/limetal02non.html]
	"""
	def __init__(self, c1=1/6, c2=30., c3=5., c4=5., c5=4., c6=-5., c7=-100., name="Lim"):
		super().__init__()
		self.name = name
		self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = c1, c2, c3, c4, c5, c6, c7
		self.dim = 2
		self.outdim = 1
		
		
		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""Generalized nonpolynomial trigonametric 2d function

		.. math::
			f(x)=a((b+cx_1\sin(dx_1))(f+e^{gx_2})+h)


		Default constant values are :math:`c = (1/6, 30., 5., 5., 4., -5., -100.)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,2)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return (self.c1 * ((self.c2 + self.c3 * x[:, 0] * np.sin(self.c4 * x[:, 0])) * (self.c5 + np.exp(self.c6 * x[:, 1])) + self.c7))[:, np.newaxis]

	def grad(self, x):
		grad = np.zeros((x.shape[0], self.outdim, self.dim))
		x1, x2 = x[:, 0], x[:, 1]

		grad[:, 0, 0] = self.c1 * (self.c3 * (np.sin(self.c4 * x1) + self.c4 * x1 * np.cos(self.c4 * x1)) * (self.c5 + np.exp(self.c6 * x2)))
		grad[:, 0, 1] = self.c1 * ((self.c2 + self.c3 * x1 * np.sin(self.c4 * x1)) * self.c6 * np.exp(self.c6 * x2))

		return grad

class DampedCosine(Function):
	"""
	Damped Cosine [https://www.sfu.ca/~ssurjano/santetal03dc.html]
	"""
	def __init__(self, c1=-1.4, c2=3.5, name="DampedCosine"):
		super().__init__()
		self.name = name
		self.c1, self.c2 = c1, c2
		self.dim = 1
		self.outdim = 1

		self.setDimDom(domain=np.ones((self.dim, 1)) * np.array([0., 1.]))

	def __call__(self, x):
		r"""A simple 1d cosine function

		.. math::
			f(x)=e^{c_1x}\cos(c_2\pi x)


		Default constant values are :math:`c = (-1.4, 3.5)`.

		Args:
			x (np.ndarray): Input array :math:`x` of size `(N,1)`.

		Returns:
			np.ndarray: Output array of size `(N,1)`.
		"""
		return np.exp(self.c1 * x) * np.cos(self.c2 * np.pi * x)

	def grad(self, x):
		return (self.c1 * self.__call__(x) - self.c2 * np.pi * np.exp(self.c1 * x) * np.sin(self.c2 * np.pi * x))[:, np.newaxis, :]