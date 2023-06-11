---
title: Java 07 | 数据结构 - 用数组构建数据列表 list
date: 2016-12-29
author: "Cong Chan"
tags: ['Software Engineer', 'Java']
---
## 列表
List
前面说到Java的数组无法更改长度，那么也就无法实现插入或者删除数组成员。Java提供了功能更丰富的数据结构 - 列表（[list](https://docs.oracle.com/javase/8/docs/api/java/util/List.html)）。所谓列表，即有序的集合（序列），用户可以精确地控制每个元素插入到列表中的哪个位置。用户可以通过整数索引（列表中的位置）来访问元素，并搜索列表中的元素（详细可进一步参考[oracle官网](https://docs.oracle.com/javase/8/docs/api/java/util/List.html)）。
<!-- more -->
这里我们尝试以java的array为基础实现一个列表，目标是实现自动扩容 (Java中的[ArrayList](https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html)不仅仅有自动扩容, 也继承了[List]的其他功能)。在探索的过程中, 可以顺带学习很多相关的内容.
使用自上而下的设计思想搭建一个框架:
先写出最基础的部分, 也就是一个构造器，前面学过了整数数组，我们直接拿来用
```java
/** Array based list.
 */
// index   0 1  2 3 4 5 6 7
// items: [6 9 -1 2 0 0 0 0 ...]
// size: 5
public class AList {
    private int[] items;
    private int size;

    /** 构造一个初始容量100的数组，初始有效数据成员为0. */
    public AList() {
        items = new int[100];
        size = 0;
    }

    /** 下面添加其他方法
    */
}
```

然后思考我们需要什么功能，把功能需求转化为实例方法instance method的形式，先把方法的外壳描绘出来，注释上该方法的功能（目的），输入值，返回值是什么之类的。具体的功能实现可以先空着，之后一步步丰富。

### 公共与私有
Public vs. Private

在上面的代码块中，可以看到 items 和 size 都被声明为 private 私有变量, 这样就只能被所在的java文件内调用.

私有变量和方法的设计初衷是服务于程序的内部功能实现, 而不是用来和外部程序(用户)进行交互的. 设置成私有, 可以避免这些变量和方法被外部程序直接调用, 避免用户通过不恰当/容易出错的方式修改某些变量. 在程序说明文档中, 一般也会明确说明程序提供什么公共变量和方法给用户调用.

因此我们这里也提供几个 public 方法让用户调用, 这样用户就能按照我们设计的方式来访问数据。分别是`getLast()` - 访问列表最后一个元素，`get(int i)`访问第`i`个元素, 和`size()`访问列表的大小.
```java
/** 程序内的方法可以访问 private 变量 */

/** 返回列表末尾的值. */
public int getLast() {
    return items[size - 1];
}

/** 返回第 i 个值 (0 是第一个). */
public int get(int i) {
    return items[i];
}

/** 返回列表元素长度. */
public int size() {
    return size;
}
```

### 泛型数组
我们不仅希望我们的列表可以存整数，也可以存其他类型的数据，可以通过泛型解决，泛型的介绍参考[这篇文章](/java-05-variable-types#通用数据类型).

泛型数组跟前面介绍的泛型示例有一个重要的语法差异：Java不允许我们创建一个通用对象的数组，原因这里不细展开。

假如我们用`Item`来标识泛型, 那么在上面的列表类中构建泛型数组时, 我们不能用`items = new Item[8];`, 而要用`items = (Item []) new Object[8];`
```java
public class AList<Item> {
    private Item[] items;
    private int size;

    /** 构造一个初始容量100的数组，初始有效数据成员为0.  */
    public AList() {
        items = (Item[]) new Object[100]; //会有编译警告, 暂时不管
        size = 0;
    }
}
```
即使这样也会产生一个编译警告，但不影响实际使用, 后面的Casting会更详细地讨论这个问题。
```java
% javac AList.java
Note: AList.java uses unchecked or unsafe operations.
Note: Recompile with -Xlint:unchecked for details.

% javac -Xlint:unchecked AList.java
AList.java:26: warning: [unchecked] unchecked cast
found   : java.lang.Object[]
required: Item[]
        items = (Item[]) new Object[100];
                         ^
1 warning
```

### 数组扩容
Resize

一个列表应该支持基本的插入和删除数据的操作，但是因为数组本身无法更改长度，所以我们就需要一个方法，在给数组插入新数据时，先检查长度容量是否足够，如果不够，那么就要增加长度。我们考虑简单的情况, 即需要在数组末尾插入或者删除数据怎么办。

插入元素：
```java
/** 把 X 插入到列表末尾. */
public void addLast(Item x) {
    /** 检查长度容量是否足够，如果不够，那么就要增加长度 */
    if (size == items.length) {
            Item[] temp = (Item[]) new Object[size + 1];
            System.arraycopy(items, 0, temp, 0, size);
            items = temp;
        }

        items[size] = x;
        size = size + 1;
}
```

创建新array并把旧数据复制过去的过程通常称为“resizing”。其实用词不当，因为数组实际上并没有改变大小，只是把小数组上的数据复制到大数组上而已。

为了让代码更易于维护，可以把上面的代码中负责大小调整的部分包装在一个独立的method中
```java
/** 改变列表容量, capacity为改变后的容量. */
private void resize(int capacity) {
    Item[] temp = (Item[]) new Object[capacity];
    System.arraycopy(items, 0, temp, 0, size);
    items = temp;
}

/** 把 X 插入到列表末尾. */
public void addLast(Item x) {
    if (size == items.length) {
        resize(size + 1);
    }

    items[size] = x;
    size = size + 1;
}
```

删除元素：
```java
/** 删去列表最后一个值，并返回该值  */
public int removeLast() {
    Item x = getLast();
    items[size - 1] = null; // 曾经引用“删除”的元素的内存地址被清空
    size = size - 1;
    return x;
}
```
事实上即使没有`items[size - 1] = null;`,也可以达到删除元素的目的.
删除对存储对象的引用, 是为了避免“loitering”。所谓 loitering，可以理解为占着茅坑不拉屎的对象，它们已经没啥用了，却还是占用着内存。如果这个对象是些几十兆的高清图片，那么就会很消耗内存。这也是为什么安卓手机越用越慢的一个原因。

当引用/内存地址丢失时，Java会销毁对象。如果我们不清空引用，那么Java将不会垃圾回收这些本来预计要删除的对象, 因为它们实际还被列表引用着。

#### 扩容效率分析
我们直觉也会感觉到，如果按照现在的设计，即每插入一个新元素，就重新复制一遍数组，这样随着数组越来越大，效率肯定会越来越差。事实上也是这样，如果数组目前长度是100个内存块，那么插入1000次，需要创建并填充大约50万个内存块（等差数列求和N(N+1)/2，101+102+...+1000 ≈ 500000）。但假如我们第一次就扩容到1000，那么就省却了很多运算消耗。可惜我们不知道用户需要插入多少数据，所以要采取其他方法-几何调整。也就是与其按照`size + FACTOR`这样的速率增加容量, 不如按照`size * RFACTOR`成倍扩容, 前者的增加速率为1, 后者为 RFACTOR, 只要设置 RFACTOR 大于1, 就能减少扩容的次数.
```java
/** 把 X 插入到列表末尾. */
public void addLast(Item x) {
    if (size == items.length) {
        resize(size * RFACTOR); //用 RFACTOR 作为因子扩容数组,
    }

    items[size] = x;
    size = size + 1;
}
```

目前我们解决了时间效率问题, 但代价是需要更大的内存空间, 也就是空间效率下降了. 假设我们插入了十亿个item，然后再删去九亿九千万个项目。在这种情况下，我们将只使用10,000,000个内存块，剩下99％完全没有使用到。

为了解决这个问题，我们可以在数组容量利用率比较低时把容量降下来. 定义利用率 R 为列表的大小除以items数组的长度。一般当R下降到小于0.25时，我们将数组的大小减半。

### 其他功能
比如排序等, 在后面介绍链表的文章中再讨论.
