---
title: Java 类的继承扩展 Extends
date: 2017-02-25
author: "Cong Chan"
tags: ['Java']
---
## 类的继承扩展
定义class之间的层次关系.
<!-- more -->
假设要构建一个`RotatingSLList`，它具有与`SLList`相同的功能，如`addFirst, size`等，但是需要额外的`rotateRight`操作将最后一项放到列表的前面，因为继承允许子类重用已经定义的类中的代码。所以让`RotatingSLList`类从`SLList`继承部分代码:
```java
public class RotatingSLList<Item> extends SLList<Item> {}
```
`RotatingSLList`"是一种"`SLList`, `extends`可以让我们继承`SLList`的原始功能，并能修改或添加其他功能。
```java
/** The rotateRight method  takes in an existing list,
    and rotates every element one spot to the right,
    moving the last item to the front of the list.
    For example, input [5, 9, 15, 22] should return [22, 5, 9, 15].
*/
public void rotateRight() {
    Item x = removeLast();
    addFirst(x);
}
```
通过`extends`, 子类**继承**父类的所有成员，成员包括：
- 所有实例和静态变量
- 所有方法
- 所有嵌套类

但注意，构造函数不继承，并且私有成员不能被子类直接访问。

**从属**和**拥有**的区别：subclass 和 superclass 是上下级从属分类，而不是拥有与被拥有的关系，不要跟 nested class 混淆。
>Interface Methods: Default methods and abstract methods in interfaces are inherited like instance methods. However, when the supertypes of a class or interface provide multiple default methods with the same signature, the Java compiler follows inheritance rules to resolve the name conflict.
-- https://docs.oracle.com/javase/tutorial/java/IandI/index.html

### Overriding
上面的例子使用父类的`removeLast()`把最后的元素直接丢弃了，但假如有一个子类`VengefulSLList`想保留被丢弃的元素呢?

子类可以**覆盖 override**父类的方法。
> 区分 Override 与 重载 overloaded：Override 的方法 signature 相同；overloaded的方法同名但不同signature。

子类可以override父类的`removeLast`, 通过添加一个实例变量来追踪所有已删除的元素.
```java
public class VengefulSLList<Item> extends SLList<Item> {
    SLList<Item> deletedItems;

    public VengefulSLList() {
        deleteItems = new SLList<Item>();
    }

    @Override
    public Item removeLast() {
        Item x = super.removeLast();
        deletedItems.addLast(x);
        return x;
    }

    /** Prints deleted items. */
   public void printLostItems() {
       deletedItems.print();
   }
}
```

继承的好处是代码得以重复利用. 坏处是因为“Is a”的关系, debugging的路径会很烦人。

但即使不包含这个`@Override`，我们仍然override了这个方法。所以从技术上来说，`@Override`并不是必须的。但是，它可以作为一个保障, 提醒编译器我们打算override此方法, 如果过程中出现问题, 编译器可以提醒。假设当我们想 override `addLast`，却不小心写成`addLsat`。此时如果不包含`@Override`，那么可能无法发现错误。如果有了`@Override`，编译器就会提示我们修复错误。

### Constructors Are Not Inherited
> Java要求所有子类的构造函数必须先调用其某一超类的构造函数。
-- https://docs.oracle.com/javase/tutorial/java/IandI/super.html

因为逻辑上，如果作为基础的超类没有构建，那么子类的构建的无从谈起。完整的子类构造函数应该是：
```java
public VengefulSLList() {
    super(); // 第一行
    deletedItems = new SLList<Item>();
}
```
之前的例子没有`super();`也可以通过编译，是因为Java会自动为我们调用超类的**无参数**构造函数。

具体分情况考虑：
1. 编译器会自动为任何没有构造函数的类提供一个无参数的默认构造函数：这个默认构造函数将调用其超类的（accessible）无参构造函数。
2. 如果子类构造函数没有指定要调用哪个超类构造函数：则编译器将自动调用超类的可访问的**无参数**构造函数
    ```java
    public class Base { }
    public class Derived extends Base { }
    ```
    - 如果其超类有**有参数**构造函数，但没有无参数构造函数，那么编译出错：
    ```java
    public class Base { public Base(String s) { } }
    ```
    此时要在子类构造函数第一行添加`super(s)`
    - 如果超类的无参数构造函数是不可访问的，那么编译出错：
    ```java
    public class Base { private Base() { } }
    ```
    - 如果没有显式的超类，那么就调用隐式的超类`Object`的无参构造函数。

Constructor Chaining：当子类构造函数调用其父类的构造函数时（无论是显式还是隐式调用），可以认为有一链式的连续调用构造函数，一直到`Object`的构造函数

同样， 可以通过`super.someMethod()`在子类中调用父类的方法

### Inheritance Cheatsheet
VengefulSLList extends SLList means VengefulSLList "is-an" SLList, and inherits all of SLList's members:
总结 Inheritance 的一些要点:
* 当子类`VengefulSLList` `extends` 超类`SLList`时, 意味着`VengefulSLList`也"是"`SLList`, 并继承`SLList`的所有成员:
    * Variables, methods, nested classes
    * 除了 constructors: Subclass constructors 必须先调用 superclass constructor; 通过 `super` 调用 overridden superclass methods 和 constructors.

调用 overridden methods 遵循两个规则:
- 编译器只允许与 static type 符合的行为.
- 对于 overridden methods, 调用是基于 dynamic type
- 可以使用 casting 来规避 compiler type checking.

### （子类型）多态
Subtype Polymorphism
> 多态（polymorphism），是指相同的消息给予不同的对象会引发不同的动作。
* 动态多态（dynamic polymorphism）：通过类继承机制和虚函数机制生效于运行期。可以优雅地处理异质对象集合，只要其共同的基类定义了虚函数的接口。
    * 在面向对象程序设计中，多态一般是指子类型多态（Subtype polymorphism）或包含多态（inclusion polymorphism）。一般是通过某种可代换性（ substitutability）与另一个数据类型（超类型，supertype）相关的数据类型，这意味着为在这个超类型的元素上运算而写计算机程序也可以在子类型的元素上运算。
* 静态多态（static polymorphism）：模板也允许将不同的特殊行为和单个泛化记号相关联，由于这种关联处理于编译期而非运行期，因此被称为“静态”。可以用来实现类型安全、运行高效的同质对象集合操作。C++ STL不采用动态多态来实现就是个例子。
    * 非参数化多态或译作特设多态（Ad-hoc polymorphism）：
        * 函数重载（Function Overloading）
        * 运算符重载（Operator Overloading）
        * 带变量的宏多态（macro polymorphism）
    * 参数化多态（Parametric polymorphism）：把类型作为参数的多态。在面向对象程序设计中，这被称作泛型编程。

> 子类型反映了类型（即面向对象的接口）之间的关系；而继承反映了一类对象可以从另一类对象创造出来，是语言特性的实现。因此，子类型也称接口继承；继承称作实现继承。

多态允许引用和对象类型不同, 如引用类型可以是实际对象类型的父类: `Animal myDog = new Dog();`. 任何通过IS-A测试的类型，任何extends过声明引用变量类型的对象都可以被赋值给该引用变量。

多态有很多应用场景，比如可以容纳不同子类型的数组：
```java
Animal[] animals = new Animal[3];
animals[0] = new Dog();
animals[1] = new Cat();
animals[2] = new Wolf();
```
参数和返回类型也可以多态. 这样通过多态, 可以编写自适应任何新类型子类的程序.

### The Object Class
Java中的每个类都是 **`Object`** class的后代，或者扩展了Object类。即使在类中没有显式的`extends`仍然隐式地继承了`Object`。也就是所有 classes 都继承了 `Object`. 既然`Object`是所有类的超类, 那为何不适用它来构造万用数据结构(或者方法)呢? 事实上的确很多ArrayList方法都用到`Object`超级类型.

`Object`类提供的方法, 都是任何对象都需要用到的方法:
```java
// -- https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html
String toString()
boolean equals(Object obj)
Class<?> getClass()
int hashCode()
protected Object clone()
protected void finalize()
void notify()
void notifyAll()
void wait()
void wait(long timeout)
void wait(long timeout, int nanos)
```
`==`检查两个变量是否引用同一个对象（检查内存地址位）; `.equals()`默认是等同于`==`, 但不同的类可能会通过 override 重写它的功能(如`Array.equals()`就是重写为检查数组内容是否相同). 当override `.equals()`时，注意：
1. 必须体现 equivalence relation
    * reflexive: x.equals(x) is true
    * symmetric: x.equals(y) IFF y.equals(x)
    * transitive: x.equals(y) and y.equals(z) implies x.equals(z)
2. 要 override 原本的`.equals()`，必须接收一个 Object 参数
3. 必须 consistent：假如`x.equals(y)`, 那么只要`x`和`y`保持不变, 那么`x`继续等于`y`.
4. `null`永远非真: `x.equals(null)` 一定是`false`

> Interfaces don’t extend Object.
-- http://docs.oracle.com/javase/specs/jls/se7/html/jls-9.html#jls-9.2

但是, 使用`Object`多态是有代价的:
```java
ArrayList<Object> dogs = new ArrayList<Object>();
Dog a = new Dog();
dogs.add(a);
// Dog b = dogs.get(0); 无法编译
```
无法编译是因为, `dogs.get()`返回的是`Object`类型, 因此编译器无法保证返回的是`Dog`.

根本原因是因为类型检查。

### 类型检查
Type Checking
```java
public class VengefulSLList<Item> extends SLList<Item> {
    ...
}

public static void main(String[] args) {
    VengefulSLList<Integer> vsl = new VengefulSLList<Integer>(9);
    SLList<Integer> sl = vsl; // 超类包含子类
    //sl dynamic type is VengefulSLList
    sl.addLast(50);
    sl.removeLast(); // 根据 dynamic type 选择 VengefulSLList 的 removeLast

    sl.printLostItems(); //编译不过, 因为编译时检查的是 static type

    VengefulSLList<Integer> vsl2 = sl; // 编译不过, 子类无法包含超类
}
```
Expressions 是 compile-time types (static), 使用`new`的表达式具有指定的 compile-time types:
- `SLList<Integer> sl = new VengefulSLList<Integer>();`, 表达式右边 compile-time types 是`VengefulSLList`。编译器检查并保证`VengefulSLList`一定也是`SLList`，因此允许此赋值.
- `VengefulSLList<Integer> vsl = new SLList<Integer>();`, 表达式右边 compile-time types 是`SLList`。因为`SLList`并不一定是`VengefulSLList`，故编译报错.

Static type checking 好处: Checks for errors early , reads more like a story

坏处就是不够灵活。

Method calls have compile-time types equal to their declared type.
```java
public static Dog maxDog(Dog d1, Dog d2) { ... }
Poodle frank = new Poodle("Frank", 5);
Poodle frankJr = new Poodle("Frank Jr.", 15);
Dog largerDog = maxDog(frank, frankJr);

// 编译不过! RHS compile-time type is Dog
Poodle largerPoodle = maxDog(frank, frankJr);
```
编译器报错, `maxDog`返回的是`Dog`, 虽然此时我们都知道这里的"狗"肯定是指贵宾犬, 但编译器无法确认`Dog`一定是`largerPoodle`. 有没有办法让编译器认可这种关系呢? 有！

### Casting
通过 casting, 可以告诉编译器一个表达式有某个特定的 compile-time types.
```java
Poodle largerPoodle = (Poodle) maxDog(frank, frankJr);
```
编译通过, 右边 compile-time type 转换为 `Poodle`.

> Caution: Casting is a powerful but dangerous tool. Essentially, casting is telling the compiler not to do its type-checking duties - telling it to trust you and act the way you want it to.

如果程序猿也无法确认类型, 可以使用`instanceof`来检查
```java
if (o instanceof Poodle)
    Poodle largerPoodle = (Poodle) maxDog(frank, frankJr);
```
