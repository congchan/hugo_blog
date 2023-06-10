title: Java 12 | 接口 Interface
date: 2017-02-23
categories:
- CS
tags:
- Java
---
子类在什么情况下需要多个父类？
<!-- more -->
比如，因为原来的业务需求比较单一，只需要`Animal - Canine - Dog`这种类结构就满足需求了, 此时`Dog`只是`Animal`的子类. 但后来有了新的功能需求, 上线了宠物功能, 理论上可以为每一种具体的属于宠物的子类添加宠物功能, 这就涉及大量的人工和bug. 但假如额外设计一种`Pet`类, 那么`Pet`和`Animal`会有交叉重叠, 如果让宠物子类同时继承两种超类, 那就是**多重继承**. 因为多重继承会有**致命方块**的问题, 不同父类对同一个方法的可能有不同的实现方式, 这会导致冲突. 所以Java不支持这种方式.

## 接口
而接口这个概念, 就可以用于解决这个问题的. 类不需要继承多个父类, 只需要实现一个或多个接口指定的所有方法/行为的关系.
```java
public interface Pet
{
    public abstract void beFriendly();
    public abstract void play();
}
// 对于属于宠物的子类，让其实现接口`Pet`.
public class Dog extends Canine implements Pet {}
```

### 接口作为参数
我们前面创建的 `LinkedList` and `ArrayList` 其实很相似 - 所有的method都一样.

如果我们需要写一个需要用到数组的类比如`WordUtils` class,
```java
public class WordUtils {
   /** Returns the length of the longest word. */
   public static String longest(LinkedList<String> list) {
      ...
      return list.get(maxDex);
   }
}
```
我们如何让`longest`方法也适配`ArrayList`？简单的方法及时写两个同名不同参数的methods。即所谓`method overloading`。
`public static String longest(LinkedList<String> list)`
`public static String longest(ArrayList<String> list)`

但 overload 有几个缺点:
* 重复冗余，写两个几乎相同的代码块。
* 产生更多需要维护的代码，那意味着如果你想对的方法做一个小优化或debug，你需要在对应每种list的方法中改变它。
* 如果我们想要适配更多的列表类型，不得不复制每个新列表类的方法。

为避免以上问题，我们一般希望能尽量把两个功能近似的方法合并，但要保证其足够广泛的适用场景。

定义通用列表接口 `interface List`。然后把`LinkedList`和`ArrayList`实现`List`。
```java
public interface List<Item> {
    public void addFirst(Item x);
    ...
}
```
这里的 List 是接口。本质上是一个指定必须能够做什么的合约，但不提供具体实现。
```java
public class ArrayList<Item> implements List<Item>{
    // 具体的执行
    public void addFirst(Item x) {
        insert(x, 0);
    }
}

public class LinkedList<Item> implements List<Item>{
    // 具体的执行
}
```
`ArrayList<Item> implements List<Item>`类似签合约 - `ArrayList`保证实现`List`接口列出的所有属性（变量）和方法。

这样就可以同时适配多种list：
```java
public class WordUtils {
   /** Returns the length of the longest word. */
   public static String longest(List<String> list) {
      ...
      return list.get(maxDex);
   }

    public static void main(String[] args) {
     ArrayList<String> someList = new ArrayList<>();
     //or
     LinkedList<String> someList = new LinkedList<>();
     ...
     System.out.println(longest(someList));
  }
}
```
接口列出所有方法的声明，就像‘合约’，但没有具体的实现. 根据‘合约’，由子类来实现且必须实现（override）每一个method，否则无法通过编译. 继承关系可以延续多代。例如，B可以继承A，C可以继承B.

### GRoE
根据Java的Golden Rule of Equals，每一个赋值a = b，本质上是把b中的bits拷贝到a中，着要求b和a的类型相同。

同理, 假设`public static String longest(List<String> list)`既接受List, 也接受ArrayList和LinkedList，但是由于ArrayList和List是不同的类，那怎么遵守GRoE呢？

因为ArrayList与List有着上下位包含的关系，这意味着ArrayList应该能够赋值给List的内存位中.
```java
public static void main(String[] args) {
    List<String> someList = new SLList<String>();
    someList.addFirst("elk");
}
```
这段代码运行时，会创建SLList并将其地址存储在someList变量中。
