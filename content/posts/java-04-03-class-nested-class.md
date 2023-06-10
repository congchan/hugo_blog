title: Java 04 | 类 class - 03 嵌套类
date: 2016-12-25
categories:
- CS
tags:
- Software Engineer
- Java
---
## 嵌套类
我们经常需要在某个类A中使用另一个类B，如果设计时我们知道类B只有在类A中有被使用的可能性, 那么我们可以把类B定义在类A中, 作为类A的嵌套类, 类A就称之为外部类. 这样做可以隐藏类名，减少全局的标识符，从而限制用户能否使用该类建立对象。这样可以提高类的抽象能力，并且强调了两个类(外围类和嵌套类)之间的主从关系。
<!-- more -->
> A nested class is any class whose declaration occurs within the body of another class or interface. A top level class is a class that is not a nested class.

嵌套类分为两类：静态和非静态。声明为static的嵌套类简称为**静态嵌套类**。非静态嵌套类称为**内部类**(inner class)。
```java
class OuterClass {
    ...
    static class StaticNestedClass {
        ...
    }
    class InnerClass {
        ...
    }
}
```
作为OuterClass的成员，嵌套类可以声明为private，public，protected或package private。外部类只能声明为public或package private。更多详情[参考官网](https://docs.oracle.com/javase/tutorial/java/javaOO/nested.html).

### 内部类
内部类可以直接访问外部类的方法和变量(即使是private的也可以)。如果从外部类程序代码中初始化内部类, 此内部对象会绑在该外部对象上. 一个内部类的实例作为成员存在于所属外部类的实例中。
```java
public class Outer {
    private int outVar;
    class Inner {
        public int inVar;
        void go() { outVar += inVar; }
    }

    Inner myInner = new Inner();

    public void do() {
        myInner.go();
    }
}

public static void main(String[] args) {
    Outer O = new Outer();
    Outer.Inner I = O.new Inner();
}
```
也可以从外部类之外的程序来初始化外部类的内部实例, 语法比较特殊
```java
class Foo {
    public static void main (String[] args) {
        Outer A = new Outer();
        Outer.Inner B = A.new Inner();
    }
}
```
因为内部类与一个实例相关联，所以它不能自己定义任何静态成员。

内部类提供在一个类中实现同一个接口的多次机会. 使用了内部类, 就可以多次实现同一个方法.

除此之外, 更重要的是, 如果外部类已经继承了一种父类, 但又需要实现其他类的行为. 如果只需要实现一次, 那么可以使用实现接口, 但如果要多次实现某种行为, 那么就需要一个内部类来创建多个内部实例, 已达成需要多次实现的行为. 而这个内部类是不太可能被其他非外部类的类使用的. 这种情况多见于GUI的使用中.

### 静态嵌套类
如果嵌套类不需要使用外部类的任何实例方法或变量，那可以声明嵌套类为static。像静态类方法一样， **静态**嵌套类不能直接引用其外部类中定义的**实例**变量或方法。外部类不能直接访问静态嵌套类的成员变量，要通过静态嵌套类来访问。
