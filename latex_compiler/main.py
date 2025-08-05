from lexer import *
from parser import *
from codegen import *
from autograd import *
from optimizer import *
from function_creator import *

import numpy as np

# (name, latex, constants, domain)
func_list = [
	("Dolan", r"\abs{(x_1 + c_1x_2)\sin(x_1) - c_2x_3 - c_3x_4\cos(x_5 + x_5 - x_1) + c_4x_5^2 - x_2 - c_5}", [1.7, 1.5, 0.1, 0.2, 1], [[-100, 100],[-100, 100],[-100, 100],[-100, 100],[-100, 100]]),
	("EggCrate", r"x_1^2 + x_2^2 + c_1 \left[ \sin^2(x_1) + \sin^2(x_2) \right]", [25.], [[-5, 5], [-5, 5]]),
	("ElAttarVidyasagarDutta", r"(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2 + (x_1^2 + x_2^3 - c_3)^2", [10., 7., 1.], [[-100, 100],[-100, 100]]),
	("FreudensteinRoth", r"\left(x_1 - c_1 + \left[(c_2 - x_2)x_2 - c_3 \right] x_2 \right)^2 + \left (x_1 - c_4 + \left[(x_2 + c_5)x_2 - c_6 \right] x_2 \right)^2", [13., 5., 2., 29., 1., 14.], [[-10, 10],[-10,10]]),
	("GoldsteinPrice", r"\left[ 1+(x_1+x_2+1)^2(19-14x_1+3x_1^2-14x_2+6x_1x_2+3x_2^2) \right] \left[ 30+(2x_1-3x_2)^2(18-32x_1+12x_1^2+48x_2-36x_1x_2+27x_2^2) \right]", [], [[-2, 2],[-2, 2]]),
	("HimmelBlau", r"(x_1^2 + x_2 - c_1)^2 + (x_1 + x_2^2 - c_2)^2", [11., 7.], [[-6, 6], [-6, 6]]),
	("Hosaki", r"\left ( c_1 - c_2x_1 + c_3x_1^2 - c_4x_1^3 + c_5x_1^4 \right )x_2^2e^{-x_1}", [1., 8., 7., 7/3, 1/4], [[0, 10],[0, 10]]),
	("Keane", r"\frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}{\sqrt{x_1^2 + x_2^2}}", [], [[0, 10],[0, 10]]),
	("Leon", r" \left(1 - x_{1}\right)^{2} + c_1 \left(x_{2} - x_{1}^{2} \right)^{2}", [100], [[-1.2, 1.2],[-1.2, 1.2]]),
	("Levy13", r"\left(x_{1} -c_1\right)^{2} \left[\sin^{2}\left(c_2 \pi x_{2}\right) + c_3\right] + \left(x_{2} -c_4\right)^{2} \left[\sin^{2}\left(c_5 \pi x_{2}\right) + c_6\right] + \sin^{2}\left(c_7 \pi x_{1}\right)", [1., 3., 1., 1., 2., 1., 3.], [[-10, 10],[-10, 10]]),
	("Matyas", r"c_1(x_1^2 + x_2^2) - c_2x_1x_2", [0.26, 0.48], [[-10, 10],[-10, 10]]),
	("McCormick", r"- x_{1} + c_1 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + c_2", [2., 1.], [[-1.5, 4],[-1.5, 4]]),
	("MieleCantrell", r"(e^{-x_1} - x_2)^4 + c_1(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8", [100.], [[-1,1],[-1,1],[-1,1],[-1,1]]),
	("Mishra03", r"\sqrt{\abs{\cos{\sqrt{\abs{x_1^2 + x_2^2}}}}} + c_1(x_1 + x_2)", [0.01], [[-10, 10],[-10, 10]]),
	("Mishra04", r"\sqrt{\abs{\sin{\sqrt{\abs{x_1^2 + x_2^2}}}}} + c_1(x_1 + x_2)", [0.01], [[-10, 10],[-10, 10]]),
	("Mishra05", r"\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + c_1(x_1 + x_2)", [0.01], [[-10, 10],[-10, 10]]),
	("Mishra06", r"-\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + c_1 \left[(x_1 -c_2)^2 + (x_2 - c_3)^2 \right]", [0.01, 1., 1.], [[-10, 10],[-10, 10]]),
	("Mishra08", r"0.001 \left[\abs{ x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3 + 11520x_1^2 - 5120x_1 + 2624 } \abs{ x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 } \right]^2", [], [[-10, 10],[-10, 10]]),
	# ("NewFunction01", r"\left \abs{ {\cos\left(\sqrt{\left\abs{{x_{1}^{2} + x_{2}}\right}}\right)} \right }^{0.5} + (x_{1} + x_{2})/c_1", [100.], [[-10, 10],[-10, 10]]),
	# ("NewFunction02", r"\left \abs{ {\sin\left(\sqrt{\abs{{x_{1}^{2} + x_{2}}}}\right)} \right }^{0.5} + (x_{1} + x_{2})/c_1", [100.], [[-10, 10],[-10, 10]]),
	("NewFunction03", r"c_1 x_{1} + c_2 x_{2} + \left[x_{1} + \sin^{2}\left[\left(\cos\left(x_{1}\right) + \cos\left(x_{2}\right)\right)^{2}\right] + \cos^{2}\left[\left(\sin\left(x_{1}\right) + \sin\left(x_{2}\right)\right)^{2}\right]\right]^{2}", [0.01, 0.1], [[-10, 10],[-10, 10]]),
	("Parsopoulos", r"\cos(x_1)^2 + \sin(x_2)^2", [], [[-5, 5],[-5, 5]]),
	# ("PenHolder", r"-e^{\left\abs{{e^{\left\abs{{- \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi} + 1}\right}} \cos\left(x_{1}\right) \cos\left(x_{2}\right)}\right}^{-1}}", [], [[-11, 11],[-11, 11]]),
	("Powell", r"(x_3+c_1x_1)^2+c_2(x_2-x_4)^2+(x_1-c_3x_2)^4+c_4(x_3-x_4)^4", [10., 5., 2., 10.], [[-4, 5],[-4, 5],[-4, 5],[-4, 5]]),
	("Price01", r"(\abs{ x_1 } - c_1)^2 + (\abs{ x_2 } - c_2)^2", [5, 5], [[-500, 500],[-500, 500]]),
	("Price02", r"c_1 + \sin^2(x_1) + \sin^2(x_2) - c_2e^{(-x_1^2 - x_2^2)}", [1., 0.1], [[-10, 10],[-10,10]]),
	("Price03", r"c_1(x_2 - x_1^2)^2 + \left[c_2(x_2 - c_3)^2 - x_1 - c_4 \right]^2", [100, 6.4, 0.5, 0.6], [[-50, 50],[-50, 50]]),
	("Price04", r"(c_1x_1^3x_2 - x_2^3)^2 + (c_2x_1 - x_2^2 + x_2)^2", [2, 6], [[-50, 50],[-50, 50]]),
	("Quadratic", r"-3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2 + 203.64x_2^2 + 182.25x_1x_2", [], [[-10, 10], [-10, 10]]),
	("RosenbrockModified", r"c_1 + c_2(x_2 - x_1^2)^2 + (c_3 - x_1)^2 - c_4 e^{-\frac{(x_1+1)^2 + (x_2 + 1)^2}{c_5}}", [74, 100, 1, 400, 0.1], [[-2, 2],[-2, 2]]),
	("RotatedEllipse01", r"c_1x_1^2 - c_2 x_1x_2 + c_3x_2^2", [7, 6 * np.sqrt(3), 13], [[-500, 500], [-500, 500]]),
	("RotatedEllipse02", r"x_1^2 - x_1x_2 + x_2^2", [], [[-500, 500], [-500, 500]]),
	("Schaffer01", r"c_1 + \frac{\sin^2 (x_1^2 + x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}", [0.5, 0.5, 1, 0.001], [[-100, 100],[-100, 100]]),
	("Schaffer02", r"c_1 + \frac{\sin^2 (x_1^2 - x_2^2)^2 - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}", [0.5, 0.5, 1, 0.001], [[-100, 100],[-100, 100]]),
	# ("Schaffer03", r"c_1 + \frac{\sin^2 \left( \cos \abs{ x_1^2 - x_2^2 } \right ) - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}", [0.5, 0.5, 1, 0.001], [[-100, 100],[-100, 100]]),
	("Schaffer04", r"c_1 + \frac{\cos^2 \left( \sin(x_1^2 - x_2^2) \right ) - c_2}{c_3 + c_4(x_1^2 + x_2^2)^2}", [0.5, 0.5, 1, 0.001], [[-100, 100],[-100, 100]]),
	("SchmidtVetters", r"\frac{c_1}{c_2 + (x_1 - x_2)^2} + \sin \left(\frac{\pi x_2 + x_3}{c_3} \right) + e^{\left(\frac{x_1+x_2}{x_2} - c_4\right)^2}", [1, 1, 2, 2], [[0,10],[0,10],[0,10]]),
	("Schwefel36", r"-x_1x_2(c_1 - c_2x_1 - c_3x_2)", [72, 2, 2], [[0,500],[0,500]]),
	("SixHumpCamel", r"c_1x_1^2+x_1x_2-c_2x_2^2-c_3x_1^4+c_4x_2^4+c_5x_1^6", [4, 4, 2.1, 4, 1/3], [[-5, 5],[-5, 5]]),
	# ("TestTubeHolder", r"- c_1 \left \abs{ {e^{\left\abs{{\cos\left(c_2 x_{1}^{2} + c_3 x_{2}^{2}\right)}\right}} \sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right }", [4, 1/200, 1/200], [[-10, 10], [-10, 10]]),
	("ThreeHumpCamel", r"c_1x_1^2 - c_2x_1^4 + \frac{x_1^6}{c_3} + x_1x_2 + x_2^2", [2, 1.05, 6], [[-5, 5],[-5, 5]]),
	("Treccani", r"x_1^4 + c_1x_1^3 + c_2x_1^2 + x_2^2", [4, 4], [[-5, 5],[-5, 5]]),
	("Trefethen", r"0.25 x_{1}^{2} + 0.25 x_{2}^{2} + e^{\sin\left(50 x_{1}\right)} - \sin\left(10 x_{1} + 10 x_{2}\right) + \sin\left(60 e^{x_{2}}\right) + \sin\left[70 \sin\left(x_{1}\right)\right] + \sin\left[\sin\left(80 x_{2}\right)\right]", [], [[-10, 10],[-10,10]]),
	("Ursem01", r"- \sin(c_1x_1 - c_2 \pi) - c_3 \cos(x_2) - c_4x_1", [2, 0.5, 3, 0.5], [[-2.5, 3], [-2, 2]]),
	("Ursem03", r"- \sin(c_1 \pi x_1 + c_2 \pi) \frac{c_3 - \abs{ x_1 }}{c_4} \frac{c_5 - \abs{ x_1 }}{c_6} - \sin(c_7 \pi x_2 + c_8 \pi) \frac{c_9 - \abs{ x_2 }}{c_{10}} \frac{c_{11} - \abs{ x_2 }}{c_{12}}", [2.2, 0.5, 2, 2, 3, 2, 2.2, 0.5, 2, 2, 3, 2], [[-2, 2], [-1.5, 1.5]]),
	("Ursem04", r"-c_1 \sin(c_2 \pi x_1 + c_3 \pi) \frac{c_4 - \sqrt{x_1^2 + x_2 ^ 2}}{c_5}", [3, 0.5, 0.5, 2, 4], [[-2, 2], [-2, 2]]),
	("UrsemWaves", r"-c_1x_1^2 + (x_2^2 - c_2x_2^2)x_1x_2 + c_3 \cos \left[ c_4x_1 - x_2^2(c_5 + x_1) \right ] \sin(c_6 \pi x_1)", [0.9, 4.5, 4.7, 2, 2, 2.5], [[-0.9, 1.2], [-1.2, 1.2]]),
	("VenterSobiezcczanskiSobieski", r"x_1^2 - c_1 \cos^2(x_1) - c_2 \cos(x_1^2/c_3) + x_2^2 - c_4 \cos^2(x_2) - c_5 \cos(x_2^2/c_6)", [100, 100, 30, 100, 100, 30], [[-50, 50], [-50, 50]]),
	("WayburnSeader01", r"(x_1^6 + x_2^4 - c_1)^2 + (c_2x_1 + x_2 - c_3)^2", [17, 2, 4], [[-5, 5], [-5, 5]]),
	("WayburnSeader02", r"\left[ c_1 - c_2(x_1 - c_3)^2 - c_4(x_2 - c_5)^2 \right]^2 + (x_2 - c_6)^2", [1.613, 4, 0.3125, 4, 1.625, 1], [[-500, 500], [-500, 500]]),
	("Wolfe", r"c_1(x_1^2 + x_2^2 - x_1x_2)^{c_2} + x_3", [4/3, 0.75], [[0, 2],[0, 2],[0, 2]]),
	("Zettl", r"c_1 x_{1} + \left(x_{1}^{2} - c_2 x_{1} + x_{2}^{2}\right)^{2}", [1/4, 2], [[-1, 5], [-1, 5]]),
	("Zirilli", r"c_1x_1^4 - c_2x_1^2 + c_3x_1 + c_4x_2^2", [0.25, 0.5, 0.1, 0.5], [[-10, 10], [-10, 10]]),
]

code = ""
for func in func_list:
	print(f"\t{func[0]}(),")
	# code += create_class(func[0], func[2], func[3], "A multimodal minimzation function", func[1])

with open("autogen.py", "w") as f:
	f.write(code)