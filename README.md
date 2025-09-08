# pytuq_funcs
A function bank with implemented gradients, specifically for https://github.com/sandialabs/pytuq

Uses kinda hacky/unreadable code.

Does NOT support variable subscripts (x_i), summations, products, anything too complicated.
Supports basically whatever is in the examples in benchmark folder.
The funcs commented out didn't work, but most did.

I wrote some of the code, then got tired and wrote code to write code.
Basically wherever the code looks obnoxious the latex compiler and autograd wrote it.

latex_compiler/main.py for usage

benchmark/benchmark.py for the functions

There is absolutely NO error checking! Instead, the code will start doing some random stuff. So please double check the latex code with a latex editor or something. Sometimes code just doesn't work, but it's rare.


