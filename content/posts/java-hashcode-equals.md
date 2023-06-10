title: Java Hash @Override equals() hashcode()
date: 2017-02-26
categories:
- CS
tags:
- Java
- Software Engineer
---
主要介绍：
1. Hashcode（哈希码）与 equals（判断相等）的关系
2. Hashcode 方法的底层实现原理
3. 开发中需要掌握的原则和方法

## HashSet, HashMap, HashTable
HashSet底层是调用HashMap. HashMap 使用hashCode和equals来进行对象比较。
<!-- more -->
拿HashSet和add()举例(其余的数据结构,和 remove, contains等方法类似): 假设HashSet里面已经有了obj1, 那么当调用HashSet.add(obj2)时:
1. if (obj1 == obj2), 那么没有必要调用 hashCode(), 已经有了这个对象, 没必要添加了
2. else, if hashCode 不同，那么可以直接添加了, 没必要进一步调用 obj1.equals(obj2) 来判断对象是否相等
3. else hashCode 相同，那么需要进一步调用obj1.equals(obj2)

下面这段代码虽然 HashSet 只存了 a 对象，但当检查是否包含 b 对象时，返回true。
```java
HashSet<String> wordSet = new HashSet<String>();
String a = "hello";
String b = "hello";
wordSet.add(a);
return wordSet.contains(b); // return true
```
根据[Javadoc for Set](https://docs.oracle.com/javase/6/docs/api/java/util/Set.html#add%28E%29).

> adds the specified element e to this set if the set contains no element e2 such that (e==null ? e2==null : e.equals(e2)).

根据[Javadoc for String.equals](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#equals-java.lang.Object-)

> Compares this string to the specified object. The result is true if and only if the argument is not null and is a String object that represents the same sequence of characters as this object.

Java的set是使用它包含的元素（对象）的 `equals()`来比较 b 和 a 的。这里 String 类的`equals()`method 是比较字符串值是否相等(准确的说，是先检查是不是引用同一个对象，再看是不是同一个类，再比较值)，而不是引用的对象是否一样，故`b.equals(a)`是 true。

同样的，remove 和 add 也会先进行类似检查。

问题是，为何 hashCode 不同，就没有进一步调用`equals()`的必要呢？因为有一个前提是代码遵守The hashCode contract。

### Hashcode and equals
在Java中，每个对象都有一个hashCode，它有时容易被人遗忘或误用。有以下三点需要注意，避免掉入常见的陷阱。

根据 The hashCode contract:

> Objects that are equal must have the same hash code within a running process.

除了字面意思，也有其他隐含的意思: 不相等的对象的hashcode也可能一样; 具有相同 hash code 的对象不一定相等.

> You must override hashCode() in every class that overrides equals(). Failure to do so will result in a violation of the general contract for Object.hashCode(), which will prevent your class from functioning properly in conjunction with all hash-based collections, including HashMap, HashSet, and Hashtable. --- Effective Java, by Joshua Bloch

根据这个contract，可以延伸出以下实践原则：

**一、 每当你 override equals 时，也要 override hashCode**
假如你需要使用不一样的equals判断标准，那么就需要重写equals。但假如仅仅重写equals，而不重写hashcode()，就可能会违背 The hashCode contract。

为什么？因为 hashCode method 需要同时适配真正使用到的 equals method 的判断标准。通过重写equals，我们重新声明了一种判断对象是否相等的标准，但原始的 hashCode method还是会将所有对象视为不同的对象。所以如果没有不重写hashcode，那么根据@Override equals 判断为相同的对象将拥有不同的hashcode（可能）。这样，即使已经有了这个object，在HashMap上调用 contains() 也会返回false。

例子：在Java的创建街道street这个类，在判断两条街道是否相同时，我们有自定义的规则 - 只要是在同一个城市，有同样的街道名，那么两个street就相等，即使他们是存放在不同内存位置的两个对象（Java 的 Object 原生的equals是根据引用的对象内存地址来比较判断的）。
```java
public class Street {
    private String name;
    private String city;
    // ...

    @Override
    public boolean equals(Object obj) {
       if (!(obj instanceof Street))
            return false;
        if (obj == this)
            return true;

        Street rhs = (Street) obj;
        return new EqualsBuilder().
            // if deriving: appendSuper(super.equals(obj)).
            append(name, rhs.name).
            append(age, rhs.city).
            isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 31). // two randomly chosen prime numbers
            // if deriving: appendSuper(super.hashCode()).
            append(name).
            append(city).
            toHashCode();
    }
}
```
如果没有重写hashCode()， 那么两个名字和所在城市一样的，但引用不同地址的street就会按照默认的 hashcode() 返回不一样的code，但是根据重写的equals(), 他们是一样的, 这样就违背了 hashCode contract。

为了安全起见，让Eclipse IDE 生成 equals 和 hashCode 函数：`Source > Generate hashCode() and equals()...`
![](https://eclipsesource.com/wp-content/uploads/2012/09/generate-hashcode-equals.png)

为了提醒自己, 还可以配置Eclipse以检测是否有违反此规则的情况，并为仅重写了equals但没重写hashCode的情况显示错误：`Preferences > Java > Compiler > Errors/Warnings, then use the quick filter to search for “hashcode”`
![](https://eclipsesource.com/wp-content/uploads/2012/09/hashcode-error-config.png)

### HashCode collisions
HashCode collisions 指两个不同的对象具有相同的hashcode这种情况, 这不是什么严重的问题. 只是会导致更多的搜索步骤，太多collisions就可能会降低系统性能

但是，如果将HashCode错误地用作对象的唯一句柄，例如将其用作Map中的key，那么有时会得到错误的对象。虽然collisions一般很少见，但却是不可避免的。例如，字符串“Aa”和“BB”产生相同的hashCode：2112. 因此衍生出第二个原则

**二、永远不要把hashcode当做key来使用**

Java中有4,294,967,296个（2<sup>32</sup>)可能的int值）。既然拥有40亿个插槽，collisions似乎几乎不可能对吧？

但事实上，也不是那么不可能。试想，一个房间里有23名随机人员。你如何估计里面有两个人生日一样的概率？很低？因为一年有365天？事实上，概率约为50％！这种现象被称为[生日问题(悖论)](http://en.wikipedia.org/wiki/Birthday_paradox)。

> 如果一个房间里有23个或23个以上的人，那么至少有两个人的生日相同的概率要大于50%。

问题的本质是"23人之中两两之间存在生日相同的概率"",而不是"其他22人与其中一个人的生日相同的概率".

类比到hashcode里，这意味着有77,163个不同的对象，collisions概率是50%（假设有一个理想的hashCode函数，将对象均匀分布在所有可用的buckets中）。

### HashCodes 会变
HashCode 不保证在不同的执行过程中总能返回相同的code。根据JavaDoc：Whenever it is invoked on the same object more than once during an execution of a Java application, the hashCode method must consistently return the same integer, provided no information used in equals comparisons on the object is modified. **This integer need not remain consistent from one execution of an application to another execution** of the same application.

这种情况并不常见，实际上，库中的某些类甚至指定了用于计算hashcode的精确公式（例如String）。对于这些类，hashcode总是相同的。但是，尽管大多数的hashCode方法提供了稳定的值，但我们不能依赖它。正如[这篇文章](http://martin.kleppmann.com/2012/06/18/java-hashcode-unsafe-for-distributed-systems.html)所指出的那样，Java库实际上在不同的进程中返回不同的hashCode值，这往往会让人们感到困惑。 Google的[Protocol Buffers](http://code.google.com/p/protobuf/)就是一个例子。 因此，您不应该在分布式应用程序中使用hash code。即使两者相等，远程对象的 hash code 也可能与本地的不同。

**三、不要在分布式应用程序中使用 hashCode**
此外，要意识到，hashCode函数的实现可能会随着版本的更改而改变。因此我们的代码最好不依赖任何特定的hash code 值。例如，你不应该使用hash code来保持某种状态，不然下次运行时，“相同”对象的hash code可能会不同。

所以最好的建议可能是：除非自己创建了基于 hashcode 算法，否则根本就不要使用 hashCode 呵呵……


### 总结
在依赖于 HashSet, HashMap, HashTable ... 等数据结构的程序中：
3. 仅重写 equals()，会导致业务出错
4. 仅重写 hashcode(), 在比较两个对象时不会强制Java忽略内存地址
3. 如果不涉及对象比较(比如仅仅是iteration), 那么不需要hashCode and/or equals

参考：
https://eclipsesource.com/blogs/2012/09/04/the-3-things-you-should-know-about-hashcode/
https://stackoverflow.com/questions/27581/what-issues-should-be-considered-when-overriding-equals-and-hashcode-in-java
