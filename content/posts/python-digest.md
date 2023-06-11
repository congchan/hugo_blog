---
title: Python Digest
date: 2018-05-08
author: "Cong Chan"
tags: ['Python', 'Programming Language']
---
What you will get from this Python digest:
1, Learn advanced python programming.
2, Learn new concepts, patterns, and methods that will expand your programming abilities, helping move you from a novice to an expert programmer.
3, Practice going from a problem description to a solution, using a series of assignments.

<!-- more -->

## [Operator](https://docs.python.org/2/library/operator.html)
### Emulating numeric types
In-place operation: One modifies the data-structure itself
 ```
 object.__iadd__(self, other)
 object.__isub__(self, other)
 object.__imul__(self, other)
 object.__imatmul__(self, other)
 object.__itruediv__(self, other)
 object.__ifloordiv__(self, other)
 object.__imod__(self, other)
 object.__ipow__(self, other[, modulo])
 object.__ilshift__(self, other)
 object.__irshift__(self, other)
 object.__iand__(self, other)
 object.__ixor__(self, other)¶
 object.__ior__(self, other)
 ```
 These methods are called to implement the augmented arithmetic assignments. These methods should attempt to do the operation in-place (modifying self) and return the result (which could be, but does not have to be, self).
 If x is an instance of a class with an `__iadd__()` method, `x += y` is equivalent to `x = operator.iadd(x, y)`
 ```
 B = np.arange(12).reshape(4,3)
 for b in B:
     b += 1
 print(B) # B will be changed
 ```

## Object oriented Programming
### [Class Name Guidline](https://www.python.org/dev/peps/pep-0008/#id41)
[underscore (_)](https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc):
• For storing the value of last expression in interpreter.
• For ignoring the specific values. (so-called “I don’t care”)
• To use as ‘Internationalization(i18n)’ or ‘Localization(l10n)’ functions.
• To separate the digits of number literal value.

To give special meanings and functions to name of variables or functions
• _single_leading_underscore: weak "internal use" indicator, declaring private variables, functions, methods and classes in a module. Anything with this convention are ignored in from module import *.
• single_trailing_underscore_: used by convention to avoid conflicts with Python keyword
• __double_leading_underscore: when naming a class attribute, invokes name mangling (inside class FooBar, __boo becomes _FooBar__boo; see [Designing for inheritance](#designing-for-inheritance))
• __double_leading_and_trailing_underscore__: "magic" objects or attributes that live in user-controlled namespaces. E.g. __init__, __import__ or __file__. Never invent such names; only use them as documented. See [Magic Attributes](#magic-attributes)


### [Designing for inheritance](https://www.python.org/dev/peps/pep-0008/#id49)
If your class is intended to be subclassed, and you have attributes that you do not want subclasses to use, consider naming them with double leading underscores and no trailing underscores. This invokes Python's name mangling algorithm, where the name of the class is mangled into the attribute name. This helps avoid attribute name collisions should subclasses inadvertently contain attributes with the same name.
• Note 1: Note that only the simple class name is used in the mangled name, so if a subclass chooses both the same class name and attribute name, you can still get name collisions.
• Note 2: Name mangling can make certain uses, such as debugging and __getattr__(), less convenient. However the name mangling algorithm is well documented and easy to perform manually.
• Note 3: Not everyone likes name mangling. Try to balance the need to avoid accidental name clashes with potential use by advanced callers.

### [Descriptor](https://docs.python.org/2/howto/descriptor.html)

### [Magic Attributes](https://docs.python.org/2/library/stdtypes.html#special-attributes)
`__init__` for initialization purpose.

[`object.__dict__`](https://docs.python.org/2/library/stdtypes.html#object.__dict__): A dictionary or other mapping object used to store an object’s (writable) attributes. Basically it contains all the attributes which describe the object under question. It can be used to alter or read the attributes.  

`__call__`

### Is Python call-by-value or call-by-reference?
Neither.
> In Python, (almost) everything is an object. What we commonly refer to as "variables" in Python are more properly called names. A variable is not an alias for a location in memory. Rather, it is simply a binding to a Python object, likewise, "assignment" is really the binding of a name to an object. Each binding has a scope that defines its visibility, usually the block in which the name originates.
-- https://jeffknupp.com/blog/2012/11/13/is-python-callbyvalue-or-callbyreference-neither/

Python实际上有两种对象。
* 一种是可变对象，表现出随时间变化的行为。可变对象的变更对与它绑定的所有名称都可见，如 Python list。
* 一种是不可变对象，值在创建后无法修改。
    * 跟java的 immutable reference类似的是 Python tuple：虽然 tuple 不可变，那仅是针对其自身所绑定固定的对象而言`tuple(list1, list2)`，但tuple包含的元素对象`list1, list2`本身有自己的可变属性.

所以Python的方法调用中,`foo(bar)`只是在`foo`的作用域内创建一个与参数`bar`的绑定。
* 如果`bar`指向可变对象，当`foo`更改时，这些更改可以在函数`foo`的作用域外可见。
* 如果`bar`指向一个不可变的对象，`foo`只能在其自身本地空间中创建一个名称`bar`并将其绑定到其他对象。

## Solving Problem
A general process to solve problem with three steps: understand, specify and design.
1, Start with a vague understanding that you refine into a formal specification of a problem. In this step you want to take inventory of the concepts you are dealing with.
2, Specify how this problem can be made amenable to being coded. What is the input and output? What output is desirable?
3, Design working code

?? ----(1 Vague Understanding)-->Formal specification of a problem ---(2 Specify)--->Amendable specification---(3 Design)--->Working Code

## Program Design and Development
### Dimensions of programming
* Correctness, Efficiency, Features, Elegance
* Each part takes time, learn to make Tradeoff:
    * During the process, generally Correctness comes first.
        * Test
    * But pursuing the 100% Correctness is not the best choice.
    * There is a balance of tradeoff, and sometimes saving some time and efforts to improving the Efficiency or adding more Features may be a better option.
    * Elegance is good for maintaining and improving the program, which means saving for the future. **Refactoring** - moving along the Elegance direction without changing the other dimensions.
        * DRY: don't repeat yourself
        * Reuse: save time and code lines, also reduce the possibility of mistake

### [Coding Style](https://docs.python.org/3.5/tutorial/controlflow.html#intermezzo-coding-style)
For Python, https://www.python.org/dev/peps/pep-0008 has emerged as the style guide that most projects adhere to; it promotes a very readable and eye-pleasing coding style. Here are the most important points extracted:
* Use 4-space indentation, and no tabs.

   4 spaces are a good compromise between small indentation (allows greater nesting depth) and large indentation (easier to read). Tabs introduce confusion, and are best left out.

* Wrap lines so that they don’t exceed 79 characters.

   This helps users with small displays and makes it possible to have several code files side-by-side on larger displays.

* Use blank lines to separate functions and classes, and larger blocks of code inside functions.

* When possible, put comments on a line of their own.

* Use [docstrings](#docstring).

* Use spaces around operators and after commas, but not directly inside bracketing constructs: `a = f(1, 2) + g(3, 4)`.

* Name your classes and functions consistently; the convention is to use CamelCase for classes and lower_case_with_underscores for functions and methods. Always use self as the name for the first method argument (see [A First Look at Classes](https://docs.python.org/3.5/tutorial/classes.html#tut-firstclasses) for more on classes and methods).

* Don’t use fancy encodings if your code is meant to be used in international environments. Python’s default, UTF-8, or even plain ASCII work best in any case.

* Likewise, don’t use non-ASCII characters in identifiers if there is only the slightest chance people speaking a different language will read or maintain the code.

### Docstring
An easy way to associate documentation with a function.
* Documentation Strings conventions
   * The first line should always be a short, concise summary of the object’s purpose.
   * The second line should be blank
   * The following lines should be one or more paragraphs describing the object’s calling conventions, its side effects, etc.
* The following Python file shows the declaration of docstrings within a Python source file:
   ```python
   """Assuming this is file mymodule.py, then this string, being thefirst statement in the file, will become the "mymodule" module'sdocstring when the file is imported."""

   class MyClass(object):
       """The class's docstring"""

       def my_method(self):
           """The method's docstring"""

   def my_function():
       """The function's docstring"""
   ```
* The following is an interactive session showing how the docstrings may be accessed:
   ```
   >>> import mymodule>>> help(mymodule)

   Assuming this is file mymodule.py then this string, being thefirst statement in the file will become the mymodule modulesdocstring when the file is imported

   >>> help(mymodule.MyClass)The class's docstring>>> help(mymodule.MyClass.my_method)The method's docstring>>> help(mymodule.my_function)The function's docstring>>>
   ```

## Test
It is important that each part of the specification gets turned into a piece of code that implements it and a test that tests it.
* Extreme values
### [assert](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)
Insert debugging assertions into a program. Assertions are not a substitute for unit tests or system tests, but rather a complement.
* [Using Assertions Effectively](http://wiki.python.org/moin/UsingAssertionsEffectively)
   * Places to consider putting assertions:
      * checking parameter types, classes, or values
      * checking data structure invariants
      * checking "can't happen" situations (duplicates in a list, contradictory state variables.)
      * after calling a function, to make sure that its return is reasonable

### Time
Track which part of the code is the bottle neck of efficiency
* \>> python -m cProfile file.py
* `import cProfile`, `cProfile.run('test()')`

## Aspect-oriented programming
* correct
* efficiency
   * [Tracking time](#time): to find out the bottle neck function or algorithm
   * Rethinking the implementation of the bottle neck
      * Fewer
      * Easier/smaller: [Divide and Conquer](https://github.com/ShootingSpace/Guide-to-Computer-Science/blob/master/Note%20-%20CS106B%20Stanford%20Programming%20Abstractions.md#divide-and-conquer)
* debugging  
Each part is done with some line of codes. Instead of mix different part of the code together, it would be better to define them as different function/class. Try to seperate them as much as possible.

## Function
There are many special and useful function implementation and control flow in python: lambda, map, filter, reduce, generator, etc..

### Lambda
λ, istead of defining function with `def` and a specific function name, Lambda provide a convinent way to define a function using its own native logic and methematical expression.
The benifits are
• A small function could be defined wihtin the same code structure without seperating out a specific `def` function
• Without bothering creating any proper funciton name for a small anonymous function.

Lambda implementation
1, Like nested function definitions, lambda functions can reference variables from the containing scope, returning a function from another function. This is often used to create **function wrappers**, such as Python's decorators.
```Python
# uses a lambda expression to return a function
>>> def make_incrementor(n):
...     return lambda x: x + n
...
>>> f = make_incrementor(42)  # f is declared as a lambda function "lambda x: x+42" with parameter n = 42
>>> f(0) # call f with x=0 to return the
42
>>> f(1)
43
```
This is like creating a compiler to save process cost: some parameters like default values or initial values are compiled into the compiler, program process these parameter only once, then this compiler as a function could be called many times with other input parameters which varies every time the compiler is being called(like user input values).

2, Pass a small function as an argument, sorting or max by an alternate key
```Python
  >>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
  >>> pairs.sort(key=lambda pair: pair[1])
  >>> pairs
  [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```
```Python
  >>> l =[('x',2),('y',4),('z',0)]
  >>> max(l, key = lambda x: x[0])
  >>> ('z', 0)
```

Lambda with logic control flow
```python
Lambda x,y: False if x<y else x+y
```

### [Filter](https://docs.python.org/2/library/functions.html#filter)
Construct a list from the elements of an iterable for which function returns **true**. If iterable is a string or a tuple, the result also has that type; otherwise it is always a list.
* `filter(function, iterable)` equals to
   * if function is `None`:  `[item for item in iterable if item] `
   * if not: `[item for item in iterable if function(item)]`
* `mult3 = filter(lambda x: x % 3 == 0, [1, 2, 3, 4, 5, 6, 7, 8, 9])` >>> `[3, 6, 9]`
* See [itertools.ifilter()](https://docs.python.org/2/library/itertools.html#itertools.ifilter) and [itertools.ifilterfalse()](https://docs.python.org/2/library/itertools.html#itertools.ifilterfalse) for iterator versions of this function, including a variation that filters for elements where the function returns false.

### [Map](https://docs.python.org/2/library/functions.html#map)
Apply function to every item of iterable and **return a list** of the results. If additional iterable arguments are passed, function must take that many arguments and is applied to the items from all iterables in parallel
```
>>> map(lambda x: x % 2, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> [1, 0, 1, 0, 1, 0, 1, 0, 1]
```

### [Reduce](https://docs.python.org/2/library/functions.html#reduce)
Apply function of two arguments cumulatively to the items of iterable, from left to right, so as to reduce the iterable to a single value.
   ```
   In [1]: reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
   Out[1]: 15 # ((((1+2)+3)+4)+5)
   In [1]: reduce(lambda a, b: '{}, {}'.format(a, b), [1, 2, 3, 4, 5, 6, 7, 8, 9])
   Out[1]: '1, 2, 3, 4, 5, 6, 7, 8, 9'
   ```

### List/Dict/Set Comprehensions
* List comprehensions: `[ s for r, s in cards if r in 'JQK' ]`
* Dictionary comprehensions: `{x: x ** 2 for x in range(5) if x % 2 == 0}`
* Set comprehensions: `{int(sqrt(x)) for x in range(30)}`
* And in general, we can have any number of for statements, if statements, more for statements, more if statements. The whole is read from left to right

### [Generator Expressions](https://docs.python.org/2/reference/expressions.html#generator-expressions)
Unlike the for loop in the list comprehensions which walk through the whole loop, generator will walk one step in the for loop if a `next()` is called.
* The advantage is
   * less indentation
   * stop the loop early
   * easier to edit
* Implementation of generator: `g = (sq(x) for x in range(10) if x%2 == 0).`  
   * The generator function is a promise, but no computation has been done yet.
   * `next(g)` to call a one-time calculation.
   * When reaching the end of for-loop in the generator, the `next(g)` comment will return a false called "StopIteration".
   * To avoid the "StopIteration" false
      * Use a outer for statement: `for xx in g: ...`
      * convert the generator to list: `list(g)`

### Generator functions
Using a yield expression in a function definition is sufficient to cause that definition to create a generator function instead of a normal function.
* [Yield expressions](https://docs.python.org/2/reference/expressions.html#yield-expressions)
   The yield expression is only used when defining a generator function, and can only be used in the body of a function definition.
* [Yield implementation](https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/index.html)
   ```python
   def ints(start, end=None):
       i = start
       while i <= end or end is None:
           yield i
           i = i + 1
   ```
   ```python
   def fab(max):
        n, a, b = 0, 0, 1
        while n < max:
             yield b
             # print b
             a, b = b, a + b
             n = n + 1
   ```

### Iterator
The true beneath `For Statemet` is **iterable**
```python
for x in itmes:
     print x
```
What the whole truth is:
```python
it = iter(items)
try:
    while True:
         x = next(it)
         print x
except StopIteration:
    pass
```
Overall, Python calls the thing that can be iterated over in a for loop an iterable. Strings and lists are examples of iterables, and so are generators.

[itertools library](https://docs.python.org/2/library/itertools.html#module-itertools) - Functions creating iterators for efficient looping.
`any(iterable)`: Return True if any element of the iterable is true. If the iterable is empty, return False.

### [Unpacking Argument Lists](https://docs.python.org/3/tutorial/controlflow.html#tut-unpacking-arguments)
The `*` operator simply unpacks the tuple or list and passes them as the positional arguments to the function.
```python
>>> list(range(3, 6))            # normal call with separate arguments
[3, 4, 5]
>>> args = [3, 6]
>>> list(range(*args))            # call with arguments unpacked from a list
[3, 4, 5]
```
### Handling different types of argument (*polymorphism)
An argument could be different type:
   `timedcalls(n,fn)`, if n is int `isinstance(n,int)`, it means controling the how many times fn was called, while n is float, it means controling the total runtime of fn called

### eval()



## [Decorator](https://www.python.org/dev/peps/pep-0318/)
Motivation: when applying a transformation to a function `def f(self): ...definition...; f = dec(f)`, it becomes less readable with longer methods. It also seems less than pythonic to name the function three times for what is conceptually a single declaration.

The solution is to place the decoration in the function's declaration:
```python
@dec
def foo(cls):
    pass
```
### [@property](https://www.programiz.com/python-programming/property)
`property(fget=None, fset=None, fdel=None, doc=None)`
A property object has three methods, getter(), setter(), and delete() to specify fget, fset and fdel at a later point.

`some_object = property(get_some_object,set_some_object)` equals to
```python
some_object = property()  # make empty property
some_object = some_object.getter(get_some_object) # assign fget
some_object = some_object.setter(set_some_object) # assign fset
```

Decorator as tools
• Debug tool: help developping, count calls times, count excecute time
• Performance: make the programme faster, such as dynamic programming algorithm
• Expressiveness: doc string, explaining funciton
• Trace: help to monitor the execution of the program, such as each level result printed with different indentation

Disable decorator: `dec = disabled`, make the decorator disabled.



## [Regular Expression](https://docs.python.org/2/library/re.html)
`import re`

[Reference: A Regular Expression Matcher](http://www.cs.princeton.edu/courses/archive/spr09/cos333/beautiful.html)

In C language, any number start with '0' is interpreted as an octal number( base-8 number system ):
'012' -> int 10; '09' -> invalid

**Special characters**
• `*` match 0 or more repetitions of the preceding character. ab* will match ‘a’, ‘ab’, or ‘a’ followed by any number of ‘b’s.
• `?` Causes the resulting RE to match 0 or 1 repetitions of the preceding RE. ab? will match either ‘a’ or ‘ab’.
• `.` (Dot) matches any single character
• `^` (Caret) Matches the start of the string
• `$` Matches the end of the string or just before the newline at the end of the string, foo matches both ‘foo’ and ‘foobar’, while the regular expression foo$ matches only ‘foo’
• `+` match 1 or more repetitions of the preceding RE. `ab+` will match ‘a’ followed by any non-zero number of ‘b’s; it will not match just ‘a’.

**Commonly used expression**
• Upper case letter `'[A-Z]'`
• Any alphanumeric character `[a-zA-Z0-9_]`
• Decimal digit `[0-9]`
• Non-digit character `[^0-9]`
• Whitespace character `[ \t\n\r\f\v]`

`search(string[, pos[, endpos]])`: Scan through string looking for a location where this regular expression produces a match, and return a corresponding MatchObject instance. Return None if no position in the string matches the pattern.

`re.findall(pattern, string, flags=0)`：Return all non-overlapping matches of pattern in string, as a list of strings.

### [String Formatting](https://docs.python.org/2.4/lib/typesseq-strings.html)
Modulo(`%`): String and Unicode objects have one unique built-in operation: the `%` operator (modulo). This is also known as the string formatting or interpolation operator. Given format `%` values (where format is a string or Unicode object), `%` conversion specifications in format are replaced with zero or more elements of values.
`%d`:	Signed integer decimal.
`%s`:	String (converts any python object using str()).
`print '%d: %s' % (1, 'animal')` >> `1: animal`


## Python data structure
### [Numpy indexing](https://docs.scipy.org/doc/numpy-dev/reference/arrays.indexing.html#indexing)
Ellipsis: The same as `...`. Special value used mostly in conjunction with extended slicing syntax for user-defined container data types. `a = [1,2,3], a[...] is actually the same as a`

None: extends one more demention by further slicing the corresponding c into smallest units.
```python
t = np.arange(27).reshape(3,3,3), #t shape is (3,3,3)
t[None,].shape # (1, 3, 3, 3)
t[...,None].shape # (3, 3, 3, 1)
t[:, None,:].shape # (3, 1, 3, 3)
t[:,:, None].shape # (3, 3, 1, 3)
```

## Reference
• [CS212 Design of Computer Program @Udacity](https://www.udacity.com/course/design-of-computer-programs--cs212), [Course Wiki](https://www.udacity.com/wiki/cs212)
>Syllabus
Lesson 1: How to think to solve problem
Lesson 2: Python features; Instrumentation
Lesson 3: Build function as tools; Define language; Grammar
Lesson 4: Dealing with Complexity Through Search
Lesson 5: Dealing with Uncertainty Through Probability

• [The Python Tutorial](https://docs.python.org/3/tutorial/)
• [Open Book Project: How to Think Like a Computer Scientist: Learning with Python](http://www.openbookproject.net/thinkcs/)
