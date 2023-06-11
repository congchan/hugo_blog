---
title: Python之奇技淫巧
date: 2017-02-22
author: "Cong Chan"
tags: ['Python', 'Programming Language']
---
FBI WARNING 这不是python入门 

<!-- more -->

## 函数
> Fundamentally, the qualities of good functions all reinforce the idea that functions are **abstractions**.

函数作为一种机制, 提供了用于抽象数值运算的模式, 使其独立于所涉及的特定值。

### 文档
> code is written only once, but often read many times.

docstring
```python
def pressure(v, t, n):
  """Compute the pressure in pascals of an ideal gas.

  Applies the ideal gas law: http://en.wikipedia.org/wiki/Ideal_gas_law

  v -- volume of gas, in cubic meters
  t -- absolute temperature in degrees kelvin
  n -- particles of gas
  """
```
```
>>> help(pressure)
```

[Python docstring guidelines](http://www.python.org/dev/peps/pep-0257/)

## 高阶函数
>  Functions that manipulate functions are called higher-order functions.

高阶函数进一步扩展一般函数，能表达通用的, 独立于其调用的特定函数的计算方案。

### Functions as Arguments
```python
def summation(n, term):
  total, k = 0, 1
  while k <= n:
      total, k = total + term(k), k + 1
  return total

def cube(x):
	return x*x*x

def sum_cubes(n):
  return summation(n, cube)
```

### Nested Definitions
> One negative consequence of this approach is that the global frame becomes cluttered with names of small functions, which must all be unique. Another problem is that we are constrained by particular function signatures.

当同一环境下，当出现需要相似功能但与已有函数的参数不同时，此时嵌套函数可以方便我们定义函数.
```python
def improve(update, close, guess=1):
  while not close(guess):
      guess = update(guess)
  return guess
```
这里的`update`只接受一个参数, 假如我们刚好有需要两个参数的
```python
def sqrt_update(x, a):
  """square root"""
  return average(x, a/x)
```
这个函数就无法传入`improve`中.

嵌套函数, 让`sqrt_update`传入参数保持一个, 同时额外能够获取其本地环境frame的其他参数
```python
def sqrt(a):
  def sqrt_update(x):
      return average(x, a/x)
  def sqrt_close(x):
      return approx_eq(x * x, a)
  return improve(sqrt_update, sqrt_close)
```
> local def statements only affect the current local frame.
> **lexical scoping**: sharing names among nested definitions

### Functions as Returned Values
```python
def compose1(f, g):
  def h(x):
      return f(g(x))
  return h
```
比如TensorFlow中常用的
```python
def model_fn_builder(...):
  """Returns `model_fn` closure."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The actual `model_fn`."""
    ...
    return ...

  return model_fn
```

### Currying
一种变换方式, 使用高阶函数将一个带有多个参数的函数转换为一个函数链，每个函数都接受一个参数。
```python
def curried_pow(x):
  def h(y):
    return pow(x, y)
  return h
```
```
>>> curried_pow(2)(3)
8
```

### Lambda Expressions
利用lambda表达式动态创建函数, 省去命名
```
     lambda            x            :          f(g(x))
"A function that    takes x    and returns     f(g(x))"
```
lambda 来由
> It may seem perverse to use lambda to introduce a procedure/function. The notation goes back to Alonzo Church, who in the 1930's started with a "hat" symbol; he wrote the square function as "ŷ . y × y". But frustrated typographers moved the hat to the left of the parameter and changed it to a capital lambda: "Λy . y × y"; from there the capital lambda was changed to lowercase, and now we see "λy . y × y" in math books and (lambda (y) (* y y)) in Lisp.
> —Peter Norvig (norvig.com/lispy2.html)

### Function Decorators
装饰器也是一种高阶函数,
```python
def trace(fn):
  def wrapped(x):
    print('-> ', fn, '(', x, ')')
    return fn(x)
  return wrapped

@trace
def triple(x):
    return 3 * x
```
以上等价于
```python
triple = trace(triple)
```
```
>>> triple(12)
->  <function triple at 0x102a39848> ( 12 )
36
```
