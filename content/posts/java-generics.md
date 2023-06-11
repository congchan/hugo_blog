---
title: Java 泛型
date: 2017-05-29
author: "Cong Chan"
tags: ['Java']
---
## 泛型
泛型意味着更好的类型安全性。主要目的是支持类型安全性的集合，让问题尽可能在编译阶段就能捉到。
<!-- more -->

### 泛型定义在类声明中
```java
public class ArrayLiat<E> extends AbstractLìst<E> implements List<E> {
    public boolean add (E o);
}
```
E代表用来创建赋予初始ArrayList的类型
```java
ArrayList<String> list = new ArrayList<String>;
```
编译器会自动把`E`看做`String`.

### 泛型方法
使用未定义在类声明的类型参数: 在返回类型之前指定泛型

`maxKey`: 返回给定`ArrayMap`中所有keys的最大值（仅在key可以比较的情况下）。假如这样写`public static K maxKey(Map61B<K, V> map) { ... }`会报错. 要将方法声明为泛型，**必须在返回类型前面指定正式的类型参数**
```java
public static <K extends Comparable<K>, V> K maxKey(Map61B<K, V> map) {
    List<K> keylist = map.keys();
    K largest = map.get(0);
    for (K k: keylist) {
        if (k.compareTo(largest)) {
            largest = k;
        }
    }
    return largest;
}
```
`K extends Comparable<K>` 保证了keys必须实现`Comparable`接口（也是一个generic接口）, 并可以与其他`K`进行比较。

这里没有使用`implement`, 而是用`extends`, 这里跟多态不同. `K extends Comparable<K>`是**type upper bounding**, 意味着`k`必须是一种`Comparable`, 但不需要具备`Comparable`的所有方法行为.

在inheritance的含义中，`extends`指为子类提供超类的能力. 在泛型范畴内, `extends`只是陈述一个事实：该类是其扩展的类的一个子类, 是加了约束, 而不是赋予能力.

### 泛型与多态
如果使用多态类定义下面的方法是没有问题的
```java
public void takeAnimals(ArrayList<Animal> animals) {
    for (Animal a : animals)
        a.eat();
}

public void go() {
    ArrayList<Animal> animals = new ArrayList<>();
    animals.add(new Dog());
    animals.add(new Cat());
    takeAnimals(animals); // 可以编译

    ArrayList<Dog> dogs = new ArrayList<>();
    dogs.add(new Dog());
    dogs.add(new Dog());
    takeAnimals(dogs); // ! 无法编译
}
```
那么在实际运行时, 如果声明为`ArrayList<Animal>`, 则不管传入的`animals`包含的是`Dog`还是`Cat`, 这个方法调用的都是`Animal`的`eat()`, 多态支持这种操作. 但如果声明为`ArrayList<Dog>`就不行, 静态类型检查不通过.

那么Java为何不允许这种情况编译呢? 反过来想, 如果可以会怎样? 假如方法换为这种
```java
public void takeAnimals(ArrayList<Animal> animals) {
    animals.add(new Cat());
}
```
可以看到会有很大问题. 在方法内部看来, 把`Cat`加到`ArrayList<Animal>`中是完全合法的, 但是对于从外部传入的参数`ArrayList<Dog>`来说, 就不合理了. 所以需要保证这种情况无法通过编译.

但如果是把`Dog[] dogs`中的元素改为`Cat`, 却可以通过编译.
```java
public void takeAnimals(ArrayList[] animals) {
    animals[0] = new Cat();
}
```
但在运行时, JVM会指出错误. 因为数组的类型是在runtime期间检查的.

要想在使用多态的情况下, 让方法自动调用子类型参数的方法, 就要使用万用字符(wildcard)
```java
public void takeAnimals(ArrayList<? extends Animal> animals) {
    for (Animal a : animals) a.eat();
}
```
使用万用字符, 编译器会组织任何可能破坏参数所指向集合的行为, 比如加入元素`animals.add(new Cat());`是无法编译通过的.

使用泛型, 也可以实现上面的方法, 就是让泛型继承父类
```java
public <T extends Animal> void takeAnimals(ArrayList<T> list);
```
这意味着`T`可以是任何一种`Animal`, 任何被声明为`Animal`或其子类的ArrayList都是合法的.

这两种方法等价, 如果需要传入多个参数, 那么只声明一次会更有效率
```java
public <T extends Animal> void takeThing(ArrayList<T> one, ArrayList<T> two);

public void takeAnimals(ArrayList<? extends Animal> one, ArrayList<? extends Animal> two);
```

融合两种方法的声明
```java
public static <T extends Comparable<? super T>> void sort(List<T> list);
```
这意味着`sort`支持任何一种实现了以`T`的父类为泛型的`Comparable`的类型.

### Autoboxing
在Java中调用包含 Generics 的class时，需要提供确切的类型参数。对于每一种 primitive type (`byte, short, int, long, float, double, boolean, char`)，必须要用其对应的 reference type (`Byte, Short, Integer, Long, Float, Double, Boolean, Character`) - 也即是 wrapper classes 作为泛型的实际类型参数。虽然声明函数和变量时必须要用 wraper classes，但在实际的数值传递中，对于 primitives 类型的数据，并不需要显式地转换为 reference types。

因为 Java 有 Autoboxing，可以隐式地在 wrapper/primitives 类型间转换. Java会自动 “box” 和 “unbox” primitive type 和其对应的 reference type 之间的值。也就是说，如果Java期望的是 wrapper classes （如Integer），假如即使接收到的是 int 这样的基本类型，Java也会“autoboxing”这种整数。
```java
public static void blah(Integer x) {
    System.out.println(x);
}
int x = 20;
blah(x); // 实际上会转换为 blah(new Integer(20))
```
反过来就是`unboxing`。

Autoboxing/Unboxing 注意事项:
* 不适用于 array 数组
* 有性能负担
* Wrapper types 比 primitive types 占用更多内存: 在大多数现代的系统里，对象的引用地址占用64位，还需要额外的64位开销用于存储动态类型等信息。 更多信息参考 [Memory usage of Java objects: general guide](https://www.javamex.com/tutorials/memory/object_memory_usage.shtml) 或 [Memory Usage Estimation in Java](http://blog.kiyanpro.com/2016/10/07/system_design/memory-usage-estimation-in-java/).

类型转换的静态方法:
* `Integer.parseInt("2")`, `Double.parseDouble("135.26")`, `new Boolean("true").booleanValue()`, 取`String`, 返回对应的primitive类型值.
* 将 primitive 主数据类型值转换为String
    * `double d = 22.2; String DoubleString = "" + d;`, `+`操作数是Java中唯一有重载过的运算符
    * `String s = Double.toString(d);`


### Widening
Java会根据需要在 primitive types 之间自动扩展.
```java
public static void blahDouble(double x) {
    System.out.println(“double: “ + x);
}
int x = 20;
blahDouble(x); //等同于 blahDouble((double) x)
```
但如果想从一个 wider type 转换为 narrower type，则必须手动 cast.
有关 widening 的更多详细信息，包括哪些类型比其他类型更 wider ，参阅[官方的Java文档](http://docs.oracle.com/javase/specs/jls/se8/html/jls-5.html)。
