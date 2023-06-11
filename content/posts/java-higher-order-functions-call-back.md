---
title: Java 高阶函数和回调
date: 2017-05-29
author: "Cong Chan"
tags: ['Java']
---
## Higher Order Functions
> A higher order function is a function that treats other functions as data.

在 Java 7 及之前的版本, memory boxes (variables) 不能包含指向 functions 的 pointers, 也就是无法给 functions 指定 types. 所以不能像Python一样直接把 function 作为参数传递到另一个 function 中。只能借用 interface：
<!-- more -->
```java
public interface IntUnaryFunction {
    int apply(int x);
}
```
```java
// 定义一个方法
public class TenX implements IntUnaryFunction {
    /* Returns ten times the argument. */
    public int apply(int x) {
        return 10 * x;
    }
}

// 高阶方法
public static int do_twice(IntUnaryFunction f, int x) {
    return f.apply(f.apply(x));
}

// 调用高阶方法
System.out.println(do_twice(new TenX(), 2));
```
在JAVA 8中，提供了很多函数式的接口。Java 8 引入` java.util.Function<T, R>`接口, 可以接受存储一个函数，`<T, R>`对应该函数的参数和返回对象

### Call Back
Java接口提供了回调(call back)的能力:
* 有时一个函数需要调用另一个尚未写好的 helper function, 这时这个 helper function 就是 `call back`。比如“排序函数”需要“比较函数”的帮助。
* 不同语言对于回调有不同的处理方式
    * Python, Perl, ML, Javascript 用函数传递 - [first-class functions, Higher Order Functions](#Higher-Order-Functions)
    * Java 选择把函数包含在一个接口中
    * C: function pointers.
    * C++: class-type functors.
    * C#: delegates.

比如Java的 Insertion Sort 可以排序任何类型的数据`Insertion.sort(a);`, `a`可以是`Double, String, java.io.File`数组. 在这里Callback就是对一个可执行代码的引用:
>・Client passes array of objects to `sort()` function.
>・The `sort()` function calls back object's `compareTo()` method as needed.
