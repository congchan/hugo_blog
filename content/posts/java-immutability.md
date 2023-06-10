title: Java Immutability
date: 2017-05-29
categories:
- CS
tags:
- Java
---
> An immutable data type is a data type whose instances cannot change in any observable way after instantiation.

比如`String`是immutable, `Array`是mutable.

Immutable 的类型: String, Integer, Double, Color, Vector, Transaction, Point2D.
Mutable: StringBuilder, Stack, Counter, Java array.
<!-- more -->
```java
public final class Vector {
    private final int N;
    private final double[] data;
    public Vector(double[] data) {
        this.N = data.length;
        this.data = new double[N];
        for (int i = 0; i < N; i++) {
        // defensive copy of mutable instance variables
            this.data[i] = data[i];
        }
   }
   ...
}
```
如何防止变量在第一次赋值后被更改：
* 可以使用`final`: 在 class constructor 里面, 或者变量初始化时, 给变量赋值一次, 之后就无法再被赋值了. **`final`的变量代表不能改变它的值, `final`的方法代表不能被覆盖, `final`的类代表不能被继承**.
* 要保证immutable不一定要使用`final`, 有时候也可以用`private`.

Immutable data types 因为属性不能改变, 优点是：
* 可以防止bugs, 并使debugging更容易
* 可以信赖对象具有某种行为/特质
* 更安全（出现攻击性代码时）
* 简化并发
* 可以放心地用作优先队列和符号表的键

缺点是需要更改属性，复制等操作时，需要给每个数据类型创建一个新对象

注意：
* 将一个引用声明为`final`并不会保证引用指向的对象是immutable. `public final ArrayDeque<String>() deque = new ArrayDeque<String>();`变量`deque`是`final`的, 仅意味着不能重新被赋值, 但其指向的数组队列对象自身还是可变的.
* 使用`Reflection API`，甚至可能对`private`变量进行更改

> Classes should be immutable unless there's a very good reason to make them mutable....  If a class cannot be made immutable, you should still limit its mutability as much as possible.
-- Effective Java, by Joshua Bloch
