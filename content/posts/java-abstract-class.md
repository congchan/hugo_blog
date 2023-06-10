title: Java 抽象类
date: 2017-02-26
categories:
- CS
tags:
- Java
---
## 抽象类
有些情况下，有些父类在实际应用中只有被继承的和产生多态的意义，而没有实例化的意义（比如抽象的`Animal`, `Canine`等，实例化这些抽象概念没有实际意义），所以不希望这种父类会被初始化。通过标记类为抽象类，编译器就知道该类不能创建实例(不能作为new实例, 但可以用于声明类型).
```java
abstract class Canine extends Animal {
    ...
}

Canine c = new Dog;
// Canine c = new Canine(); 无法编译
```
<!-- more -->
反之，不抽象的类就是具体类.

有抽象类自然就有抽象方法，抽象的方法没有实体，所有就不会含有具体的实现`public abstract void doSomething();`. 只有抽象类才能拥有抽象方法. 在抽象父类中定义可继承的抽象方法, 可以定义出一组子类共同的协议, 这样能够保证多态. 但因为抽象方法只是为了标记处多态而存在, 它们没有具体的内容, 这样在继承树结构下的第一个具体类就必须要实现所有的抽象方法. 当然, 树结构中的抽象类也可以提前把抽象方法实现了(default方法).

### 抽象类和接口的比较
接口是抽象类的极端形式，接口是完全抽象的，不包含实现（新的特性支持default method）。

有很多情景需要某个类继承多种父类。比如，因为原来的业务需求比较单一，只需要`Animal - Canine - Dog`这种类结构就满足需求了, 此时`Dog`只是`Animal`的子类. 但后来有了新的功能需求, 上线了宠物功能, 理论上可以为每一种具体的属于宠物的子类添加宠物功能, 这就涉及大量的人工和bug. 但假如额外设计一种`Pet`类, 那么`Pet`和`Animal`会有交叉重叠, 如果让宠物子类同时继承两种超类, 那就是**多重继承**. 因为多重继承会有**致命方块**的问题. 所以Java不支持这种方式.

而接口这个概念, 就是用于解决这个问题的:
```java
public interface Pet
{
    public abstract void beFriendly();
    public abstract void play();
}
// 对于属于宠物的子类，让其实现接口`Pet`.
public class Dog extends Canine implements Pet {}
```
基本上，接口能做的抽象类都可以做。 但接口的最大意义就是其无比的适用性, 若以接口取代具体的子类或抽象类作为参数或返回类型, 那么就可以传入任何有实现该接口的东西. 有了接口,
* 类就可以继承超过一个以上的来源: 可以继承某个父类, 并且实现其他接口.
* 接口自身可以**extends**多个其他接口.
* 一个类, 可以实现多个接口. `class Dog extends Animal implements Pet, Saveable, paintable {}`
* 一个接口, 可以给不同的类使用, 因此就可以为不同的需求组合出不同的继承层次.

等于说有了接口, 那么不管一个类是什么类, 只要它实现了一个接口, 那么就知道它一定会履行对应的行为合约. 允许不同继承树的类实现共同的接口对Java API来说是非常重要的, 比如要将对象状态保存起来, 只要去实现`Serializable`接口即可; 打算让对象以单独的线程来执行, 就实现`Runnable`.

要注意，接口能**扩展**`extends`多个接口，不能实现`implement`任何接口.

实际应用中，抽象类通常用于部分地实现接口，在接口和实际的类中间扮演一个中间概念。
```java
public interface Car {
   void move(Speed x);
   void stop();
}

public abstract class DeluxeModel implements Car {
	public double x;
	...
	public void move(Speed x) { ... }
	public abstract void autoPilot();
}

// 实现时, 要 override 所有抽象方法
public class TeslaX extends DeluxeModel {
	public void stop() { ... }
	public void autoPilot() { ... }
}
```
> 若不确定用哪种, 就优先考虑接口，以降低复杂性。
-- https://docs.oracle.com/javase/tutorial/java/IandI/abstract.html

Abstract classes 介于接口和 classes 之间。
* 方法可以是`public`或`private`, 也支持`protected`和`package private`.
* 支持任何类型的变量
* 无法实例化
* 除非指定为`abstract`，否则方法默认是具体的实现
* 每个类只能实现一个 Abstract classes
* 抽象类不需要实现其继承的接口所有抽象方法

Interface:
* 除非指定`access modifier`, 否则所有的方法默认都是`public` （Java 9 支持 `private`）
* 可以提供变量, 但都是`public static final`, 也即没有实例变量
* 无法实例化
* 除非指定为`default`，否则所有方法均为抽象的
* 一个类可以实现多个接口
* 根据协议承诺, 实现类必须实现其继承的接口的所有抽象方法; 否则要声明为抽象类.

如何判断应该设计类，子类，抽象类，还是接口呢？
* 如果新的类无法对其他的类通过IS-A测试时，就设计不继承其他类的类
* 只有在需要某类的特殊化版本时，以覆盖或增加新的方法来继承现有的类
* 当需要定义一群子类的模板，又不想让程序员初始化此模板时，设计出抽象的类给他们用
* 如果想要定义类可以扮演的角色，使用接口
