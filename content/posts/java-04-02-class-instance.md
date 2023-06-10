title: Java 04 | 类 class - 02 类与实例
date: 2016-12-24
categories:
- CS
tags:
- Software Engineer
- Java
---
## Class
类的方法和变量细分为静态的和非静态的. 静态就是可以被类调用，所以静态方法/变量也称之为类方法/变量；非静态只能由实例调用，所以也称之为实例方法/变量。
<!-- more -->

### 静态变量
类变量 Class Variables 有`static`声明(静态变量).

静态变量一般是类本身固有的属性, 被该类所有实例共享。例如，我们可能需要用狗类的另一种生物学的统称“犬科”来作为类的说明， 这个时候可以用`public static String binomen = "犬科";`，这个变量理论上是由类来访问的。静态方法也类似.

以下代码定义了一个类来模拟狗，包含一个类变量作为这个类的说明，一个类方法用于发出叫声：
```java
public class Dog {

    public static String instruction = "狗类实例"; //类变量, 说明

    public static void makeNoise() {
        System.out.println("汪!");
    }
}
```
这里没有定义`main()`, 在这种情况下如何直接运行这个类(`java Dog`), 程序是会报错的
```java
错误: 在类 Dog 中找不到 main 方法, 请将 main 方法定义为:
   public static void main(String[] args)
否则 JavaFX 应用程序类必须扩展javafx.application.Application.
```
你可以选择在里面添加一个`main()`方法. 但这次我们选择不定义具体的`main()`. 具体要如何运行, 我们可以另写一个类定义一个main()方法来调用这个类.
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog.makeNoise();
    }
}
```
类变量和方法是有局限性的。现实世界中, 并不是所有的狗都是一样的特征，仅仅靠类这个概念是无法区分不同个体的狗, 除非你为不同的狗定义不同的类（以及里面的变量和方法）, 那么就会很繁琐痛苦.

### 静态变量初始化
静态变量会在该类的任何对象创建之前就完成了初始化.
静态变量会在该类的任何静态方法执行之前就初始化.

primitive主数据类型整数默认值是`0`, 浮点数默认值是`0.0`, boolean默认值是`false`, 对象引用是`null`.

静态的`final`变量是常数: 一个被标记`final`的变量代表一旦被初始化之后就不会改动. 常数变量的名称全部都要大写.

静态final变量必须初始化后才能使用, 初始化有两种方法:
1. 声明时`public class Foo { public static final int FOO_X = 25; }`
2. 使用静态初始化程序
    ```java
    public class Bar {
        public static final double BAR_SIGN;
        static { BAR_SIGN = (double) Math.random(); }
    }
    ```


### 实例
包括实例变量Instance Variables 和实例方法Instance method.

Java的类定义就像定义一张蓝图, 我们可以在这个蓝图的基础上, 生成不同的实例instance. 实例是概念性的说法，本质上在Java里就是对象object。这样的特性提供了一个很自然而然地在java中模拟生成实体世界的方法：定义一个狗的类，在这个类的基础上，通过不同的特征参数实例化不同特征的狗（instances），并使类方法的输出取决于特定实例的狗的属性。
```java
/** 一只狗的类:*/
public class Dog {
    public int weight;

    public void makeNoise() {
        if (weight < 10) {
            System.out.println("嘤嘤嘤!");
        } else if (weight < 30) {
            System.out.println("汪汪汪");
        } else {
            System.out.println("嗷呜!");
        }
    }
}
```
这里的方法和变量没有`static`, 所以是实例（非静态）方法和变量. 如果直接用 `Dog` 类来调用这些方法, 会报错:
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog.weight = 21;
        Dog.makeNoise();
    }
}
```
```java
DogLauncher.java:3: 错误: 无法从静态上下文中引用非静态 变量 weight
        Dog.weight = 21;
           ^
DogLauncher.java:4: 错误: 无法从静态上下文中引用非静态 方法 makeNoise()
        Dog.makeNoise();
           ^
```
这个时候, 你需要实例化一只狗, 让这个实例来调用非静态变量和方法:
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog bigDog = new Dog();
        bigDog.weight = 5;
        bigDog.makeNoise();
    }
}
```
运行时，这个程序将会创建一个重量为5的狗，这个狗就会“嗷呜”叫。

虽然Java在技术上允许使用实例变量来访问静态变量或静态方法，但合法的不一定是好的, 这样会产生容易误解的代码，所以还是少用为好。

总的来说，之所以需要实例方法和变量，是因为我们需要模拟个体，一只具体的狗，并让它发出声音。这个`weight`和`makeNoise()`只能由具体的狗调用。狗类不能调用，也没有调用的意义, 毕竟每只狗的重量和声音都不同的. 在设计程序时, 如果其中一个方法我们只打算让特定的实例来调用它(而不让类去调用它), 那么这个方法应该设计成实例方法。

### 静态方法与实例方法
类(静态)方法Class Methods由类调用`Dog.makeNoise();`.
* **静态方法无法调用实例变量**. 否则会报错`non-static variable size cannot be referenced from a static context`.
* **静态方法不能调用非静态的方法**: 即使非静态方法中没有涉及实例变量, 也无法通过编译, 因为不能保证整个非静态方法在以后会不会改动涉及实例变量, 或者如果子类去覆盖此方法可能会用到实例变量.

实例方法Instance Methods只能由实例来调用`bigDog.makeNoise();`.
* **实例方法访问本成员变量是不受限制的**, 也就是它可以访问静态变量和静态方法.

可以看到实例方法更具体, 更贴近实体世界, 那我们仍需要静态方法, 因为:
* 有些类不需要实例化, 毕竟我们也经常需要处理抽象的概念, 这些抽象概念在人类认知范畴内是统一的, 比如Java的`Math`包含很多数学运算的静态方法, 这是客观规律，比如`x = Math.sqrt(100)`不管创建多少个实例，数学运算的结果都是一样的, 所以没必要实例化来浪费空间.
* 有些类有静态方法, 是有实际作用的 - 每个实例都通用的方法。若想比较一个类里面的不同实例, 比如两只狗的重量。比较简单的方法就是使用一个比较狗的重量的类方法:
```java
public static Dog maxDog(Dog d1, Dog d2) {
    if (d1.weight > d2.weight) {
        return d1;
    }
    return d2;
}
Dog d = new Dog(15);
Dog d2 = new Dog(100);
Dog.maxDog(d, d2);
```
    * 这个时候, 若使用实例方法也可以, 但没那么直观：
    ```java
    /** 我们使用关键字this来引用当前对象d。*/
    public Dog maxDog(Dog d2) {
        if (this.weight > d2.weight) {
            return this;
        }
        return d2;
    }
    Dog d = new Dog(15);
    Dog d2 = new Dog(100);
    d.maxDog(d, d2);
    ```

如果一个类只有静态的方法, 可以将构造函数标记为`private`以避免被初始化, 就像Java的`Math`一样.
