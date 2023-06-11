---
title: Java 05 | 数据类型
date: 2016-12-26
author: "Cong Chan"
tags: ['Software Engineer', 'Java']
---
## 数据类型
[数据类型](https://zh.wikibooks.org/zh-hans/Java/%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B)是程序设计语言描述事物、对象的方法。Java数据类型分为基本类型（内置类型）和引用类型(扩展类型）两大类。基本类型就是Java语言本身提供的基本数据类型，比如，整型数，浮点数，字符，布尔值等等。而引用类型则是Java语言根据基本类型扩展出的其他类型，Java要求所有的引用扩展类型都必须包括在类定义里面，这就是Java为什么是面向对象编程语言的原因...
<!-- more -->
上面的定义有点抽象，要理解数据类型，需要先理解一个问题: 神秘的海象问题

尝试预测下面的代码运行时会发生什么。b的变化是否会影响a？提示：类似Python。
```java
Walrus a = new Walrus(1000, 8.3);
Walrus b;
b = a;
b.weight = 5;
System.out.println(a);
System.out.println(b);
```
同样尝试预测下面的代码运行时会发生什么。x的改变是否影响y？
```java
int x = 5;
int y;
y = x;
x = 2;
System.out.println("x is: " + x);
System.out.println("y is: " + y);
```
答案是b的变化会影响a, 但x的改变不影响y，具体见[可视化过程](http://cscircles.cemc.uwaterloo.ca/java_visualize/#code=public+class+PollQuestions+%7B%0A+++public+static+void+main%28String%5B%5D+args%29+%7B%0A++++++Walrus+a+%3D+new+Walrus%281000,+8.3%29%3B%0A++++++Walrus+b%3B%0A++++++b+%3D+a%3B%0A++++++b.weight+%3D+5%3B%0A++++++System.out.println%28a%29%3B%0A++++++System.out.println%28b%29%3B++++++%0A%0A++++++int+x+%3D+5%3B%0A++++++int+y%3B%0A++++++y+%3D+x%3B%0A++++++x+%3D+2%3B%0A++++++System.out.println%28%22x+is%3A+%22+%2B+x%29%3B%0A++++++System.out.println%28%22y+is%3A+%22+%2B+y%29%3B++++++%0A+++%7D%0A+++%0A+++public+static+class+Walrus+%7B%0A++++++public+int+weight%3B%0A++++++public+double+tuskSize%3B%0A++++++%0A++++++public+Walrus%28int+w,+double+ts%29+%7B%0A+++++++++weight+%3D+w%3B%0A+++++++++tuskSize+%3D+ts%3B%0A++++++%7D%0A%0A++++++public+String+toString%28%29+%7B%0A+++++++++return+String.format%28%22weight%3A+%25d,+tusk+size%3A+%25.2f%22,+weight,+tuskSize%29%3B%0A++++++%7D%0A+++%7D%0A%7D&mode=edit).
这里的差别虽然微妙, 但其背后的原理对于数据结构的效率来说是非常重要的，对这个问题的深入理解也将引导我们写出更安全，更可靠的代码。

### 基本类型
Primative Types

计算机中的所有信息都以一系列1和0的形式存储在内存中，这些二进制的0和1就是比特位（bits）。比如72和“H”在内存一般以01001000的形式存储，对他们的形式是一样的。一个引申问题就是：Java代码如何解释01001000，怎么知道应该解释为72还是“H”？ 通过类型types，预先定义好类型即可, 以下代码
```java
char x = 'H';
int y = x;
System.out.println(x);
System.out.println(y);
```
会分别得到“H”和72. 在这种情况下，x和y变量都包含几乎相同的bits，但是Java解释器在输出时对它们进行了不同的处理。

Java有8种基本类型：byte，short，int，long，float，double，boolean和char。


#### 变量声明
Declaring Variables

计算机的内存可以视为包含大量用于存储信息的内存比特位，每个位都有一个唯一的地址。现代计算机可以使用许多这样的位。 当你声明一个特定类型的变量时，Java会用一串**连续**的内存位存储它。例如，如果你声明一个int，你会得到一个长度32的内存list，里面有32bits。Java中的每个数据类型都有不同的比特数。

除了留出内存空间外，Java解释器还会在一个内部表中创建一个条目，将每个变量名称映射到内存块中第一个位置（表头list head）。 例如，如果声明了int x和double y，那么Java可能会决定使用计算机内存的352到384位来存储x，而20800到20864位则用来存储y。然后解释器将记录int x从352开始，y从20800开始。

在Java语言里无法知道变量的具体内存位置，例如你不能以某种方式发现x在位置352。不像C++这样的语言，可以获取一段数据的确切地址。Java的这个特性是一个折衷！隐藏内存位置自然意味着程序猿的控制权更少，就无法做[某些类型的优化](http://www.informit.com/articles/article.aspx?p=2246428&seqNum=5)。但是，它也避免了一大类非常棘手的[编程错误](http://www.informit.com/articles/article.aspx?p=2246428&seqNum=1)。在现在计算成本如此低廉的时代，不成熟的优化还不如少点bug。

当声明一个变量时，Java不会在预留的内存位置中写入任何内容, 也即没有默认值。因此，如果没有赋值, Java编译器会阻止你使用变量。

以上只是内存分配的简要说明, 堆和栈的介绍可以参考 CS106B 笔记。


### 引用类型
Reference Types

所有基本数据类型之外的类型都是引用类型。
引用类型顾名思义，就是对对象的引用。在java中内存位置是不开放给程序员的, 但我们可以通过引用类型访问内存中某处对象。所有引用类型都是 java.lang.Object 类型的子类。
#### 对象实例化
Object Instantiation

对象实例化：当我们使用new（例 new Dog）实例化对象时，Java首先为类的每个实例变量分配一串长度合适的bits位，并用缺省值填充它们。然后，构造函数通常（但不总是）用其他值填充每个位置.
```java
public static class Walrus {
    public int weight;
    public double tuskSize;

    public Walrus(int w, double ts) {
          weight = w;
          tuskSize = ts;
    }
}
```
用`new Walrus(1000, 8.3)`创建一个Walrus实例后, 我们得到分别由一个32位(int weight = 1000)和一个64位(double tuskSize = 8.3)的内存块组成的实例：
通过程序[可视化过程](http://cscircles.cemc.uwaterloo.ca/java_visualize/#code=public+class+PollQuestions+%7B%0A+++public+static+void+main%28String%5B%5D+args%29+%7B%0A++++++Walrus+a+%3D+new+Walrus%281000,+8.3%29%3B%0A++++++Walrus+b%3B%0A++++++b+%3D+a%3B%0A++++++b.weight+%3D+5%3B%0A++++++System.out.println%28a%29%3B%0A++++++System.out.println%28b%29%3B++++++%0A%0A++++++int+x+%3D+5%3B%0A++++++int+y%3B%0A++++++y+%3D+x%3B%0A++++++x+%3D+2%3B%0A++++++System.out.println%28%22x+is%3A+%22+%2B+x%29%3B%0A++++++System.out.println%28%22y+is%3A+%22+%2B+y%29%3B++++++%0A+++%7D%0A+++%0A+++public+static+class+Walrus+%7B%0A++++++public+int+weight%3B%0A++++++public+double+tuskSize%3B%0A++++++%0A++++++public+Walrus%28int+w,+double+ts%29+%7B%0A+++++++++weight+%3D+w%3B%0A+++++++++tuskSize+%3D+ts%3B%0A++++++%7D%0A%0A++++++public+String+toString%28%29+%7B%0A+++++++++return+String.format%28%22weight%3A+%25d,+tusk+size%3A+%25.2f%22,+weight,+tuskSize%29%3B%0A++++++%7D%0A+++%7D%0A%7D&mode=edit))来更好地理解. 当然在Java编程语言的实际实现中，实例化对象时都有一些额外的内存开销, 这里不展开.

通过 new 实例化对象，new 会返回该对象的内存地址给我们，但假如我们没有用一个变量去接收这个地址，那么我们就无法访问这个对象。之后该对象会被作为垃圾回收。

#### 引用变量声明
Reference Variable Declaration

前面有提到，我们需要声明变量来接受实例化的对象在内存中的地址。当声明任何引用类型的变量（比如array, 前面的Dog类等）时，Java都会分配一串64位的内存位置. 这个64位的内存块仅用于记录变量的内存**地址**, 所谓内存地址, 可以理解为内存(房子)的编号(地址), 一般是内存块的表头位置的64位表达式
```java
Walrus someWalrus; // 创建一个64位的内存位置
someWalrus = new Walrus(1000, 8.3); //创建一个新的实例
/** 内存地址由 new 返回, 并被复制/赋值给 someWalrus 对应的内存位置
*/
```
比如, 假设`weight`是从内存位5051956592385990207开始存储的，后面连续跟着其他实例变量，那么就可以把5051956592385990207存储在`someWalrus`变量中。5051956592385990207由64位的二进制0100011000011100001001111100000100011101110111000001111000111111表达，这样`someWalrus`的内存就可以抽象的理解为一个表
`someWalrus: 0100011000011100001001111100000100011101110111000001111000111111 -> 具体存放实例的内存(Walrus: weight=1000, tuskSize=8.3)`
'->'可以理解为指针.

前面有提到，如果丢失了引用变量存储的内存地址，那么该地址对应的对象就找不回来了。例如，如果一个特定的 Walrus 地址的唯一副本存储在x中，那么`x = null`这行代码将删去地址，我们则丢失了这个 Walrus 对象。这也不一定是坏事，很多时候在完成了一个对象后就不在需要了，只需简单地丢弃这个参考地址就可以了。

### 等值规则
Java Rule of Equals

**对于`y = x`，Java解释器会将x的位拷贝到y中**,这个规则适用于java中任何使用`=`赋值的语法, 是理解开头的"神秘的海象"问题的关键.
* 基本类型变量的位, 存储赋值的值（基本类型）在内存中值(具体位数取决于具体的类型)
```java
int x = 5; // 此时是把内存中的某一个地址 p 复制给 x
int y;
y = x; // y 也指向 p
x = 2; // 把一个新的内存地址 new p 复制给x, 但y还是指向原来的p
```
    x的位存储的是基本类型`int 5`(32 bits), `x = 2`是把新的基本类型`int 2`复制给x, 但y还是指向原来的`int 5`， 所以y没变化。
* 引用类型 reference type 变量的位, 存储赋值的值（引用类型）在内存中的地址(固定的64 bits)
```java
Dog a = new Dog(5); // 创建一个64位的内存位, 并赋值一个新的实例 p
Dog b; // 仅创建一个64位的内存位, 没有引用内存地址(null)
b = a; // 把a的位（是实例 p 的内存地址）复制给b, 这样 b 也是指向实例 p
b.weight = 21; // 此时修改b, 会改写b指向的内存实例 p
```
    a和b只存储地址, 而它们的地址都指向相同的实例；
    如果对 b 的修改本质是对 p的修改, 那么输出`a.weight`的时候, 就会变成`21`.

### 参数传递
Parameter Passing

给函数传递参数，本质上也是赋值操作，参考上面的等值规则，也即复制这些参数的bits给函数，也称之为pass by value。Java的参数传递都是pass by value。至于传递过去的参数会不会因为函数内部的操作而更改，其判断原理在上面的[等值规则](#等值规则)已经阐明。

### 通用数据类型
Generic

在定义类的时候，有时候我们可能希望这个类能够接受任何类型的数据，而不仅仅是限定了基本类型中的任何一种。比如我们想实现一个类似excel表格的类，自然需要这个表格类能够接收各种类型的字符，数字，并呈现出来。这个时候就需要使用泛型 Generic, 也即通用数据类型。

> Guiding principles. Welcome compile-time errors; avoid run-time errors.

在2004年，Java的设计者在语言中加入了泛型，使​​我们能够创建包含任何引用类型的数据结构。方法就是在类声明的类名后面，使用一个任意的占位符，并用尖括号括住`<随便什么字符>`。然后，在任何你想使用泛型的地方，改用占位符。
```java
public class table {
    public class table {
        public int item;
        ...
    }
    ...
}
```
改为
```java
public class table<xxx> {
    public class table {
        public xxx item;
        ...
    }
    ...
}
```
`<xxx>`里面的名称并不重要, 改成其他也行, 只是一个标识符, 用来接受参数, 当用户实例化这个类时, 必须使用特殊的语法`table<String> d = new table<>("hello");`

由于泛型仅适用于引用类型，因此我们不能将基本类型`int`等放在尖括号内。相反，我们使用基本类型的引用版本，比如对于int, 用 Integer，`table<Integer> d = new table<>("10");`

总结使用方法:
* 在一个实现某数据结构的`.java`文件中，在类名后面, 只指定泛型类型一次。
* 在其他使用该数据结构的java文件中，声明实例变量时要指定所需的类型。
* 如果您需要在基本类型上实例化泛型，请使用`Integer, Double, Character, Boolean, Long, Short, Byte, Float`，而不是其基本类型。
