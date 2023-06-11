---
title: Java 13 | 接口默认方法
date: 2017-02-24
author: "Cong Chan"
tags: ['Java']
---
除了单纯提供声明之外，Java 8 也允许接口提供具体的实现方法。
<!-- more -->

### 缺省方法
从 Java 8开始支持 Default method。

我们可以在List中列出已实现的method。这些方法就是 default method，定义了List hypernyms的一些默认行为：`default public void method() { ... }`.

我们可以自由调用interface中定义的方法，而不用操心具体的实现。Default method 适用于实现接口的任何类型的对象！子类可以直接调用，而不必重新实现 default method。
```java
// List
default public void print() {
    ...
}
```
不过，我们仍然可以override default method，在子类中重新定义该方法。这样，只要我们在LinkedLList上调用`print()`，它就会调用子类override的方案，而不是父类的。
```java
// LinkedList
@Override
public void print() {
    ...
}
```

### Dynamic type
Java是通过一个叫“dynamic method selection”的特性，来确定要调用 default method 还是已经被子类override的method。

当实例声明`List<String> l = new LinkedList<String>();`, 则指明`l`是 static 类型的 `List`。由 new 生成的 object 是LinkedList类型，也从属于 List 类型。但是，因为这个对象本身是使用 LinkedList 构造函数实例化的，所以我们称之为 dynamic type。
> Dynamic type 的名称源于: 当`l`被重新分配指向另一种类型的对象时，比如说一个 ArrayList 对象，`l`的动态类型现在就变为 ArrayList. 因为它根据当前引用的对象的类型而改变, 所以是动态的。

Static vs. Dynamic Type:
* Java 每个变量都有一个static type （compile-time type），这是变量声明时指定的类型，在编译时会检查。
* 每个变量也有一个 Dynamic Type（run-time type），此类型在变量实例化（new）时指定，并在运行时检查。等同于地址指向的对象的类型。

当Java运行一个被overriden的方法时，会根据该实例的dynamic type 匹配对应的 method。

注意，如果是overload:
```java
public static void peek(List<String> list) {
    ...
}
public static void peek(LinkedList<String> list) {
    ...
}
```
对于上面的实例化的`l`, 当Java检查要调用哪个方法时，它会检查 static type (此时是List)并使用相同类型的参数调用该方法，也就是使用List作为签名的那个方法。

Implementation inheritance 也有一些缺点：
* 人会犯错。我们有可能忘了自己曾经override过一个方法。
* 如果两个接口给出冲突的 default method，则可能很难解决冲突。
* 无形中鼓励代码复杂化。
* Breaks encapsulation!
