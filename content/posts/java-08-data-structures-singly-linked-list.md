title: Java 08 | 数据结构 - 单向链表 Singly Linked List
date: 2017-01-12
categories:
- CS
tags:
- Software Engineer
- Java
---
## 链表
Linked List

前面有介绍以array为基础搭建的列表，支持自动扩容, 各种插入，删除速度都很快. 这里再介绍另一种方案, 链表, 也可以实现列表自动扩容.

### 带链接的节点
链表的核心组成是带链接的节点, 每个节点就像火车车厢, 有钩子连接下一节车厢.![](/images/408px-Singly-linked-list.png)
<!-- more -->
以int节点为例:
```java
public class IntNode {
    public int item;
    public IntNode next;

    public IntNode(int i, IntNode n) {
        item = i;
        next = n;
    }
}
```
`next`就是这个链接, 每一个节点就是其上一个节点的`next`.

### 使用嵌套类
这个节点作为一个相对独立的数据结构, 我们更希望让他单独作为一个类来维护. 再另外创建一个名为`LinkedList`的class与用户进行交互. 这样还有另一个好处就是提供一个命名为`LinkedList`的类给用户交互，用户更直观地知道自己是在调用链表。如果直接与node类交互，用户可能会困扰. 但同时考虑到这个node类只有`LinkedList`会调用，所以我们可以把node类嵌套进`LinkedList`中。
```java
public class LinkedList<XXX> {
    private class OneNode {
        public XXX item;
        public OneNode next;

        public OneNode(XXX i, OneNode n) {
            item = i;
            next = n;
        }
    }

    private OneNode first;
    private int size;

    public LinkedList(XXX x) {
        first = new OneNode(x, null);
        size = 1;
    }
    //下面是各种方法...
}
```
以上定义使用了[泛型](/java-05-variable-types#通用数据类型)。声明`OneNode`实例`first`为私有变量, 是为了防止用户错误地摆弄链接指向，`private`和`public`的使用[参考](/java-07-data-structures-array-based-list#公共与私有).

### 补充必要的实例方法
插入的操作核心是改变链接指向， 比如原来是`A->B->D`, 要插入C, 则把`C.next`指向D,然后把`B.next`改为指向C, 变为`A->B->C->D`
```java
public class LinkedList<XXX> {
    private class OneNode { ... }

    private OneNode first;
    private int size;

    public LinkedList(XXX x) { ... }
    /** 在列表开头插入 x. */
    public void addFirst(XXX x) {
        first = new OneNode(x, first);
        size += 1;
    }

    /** 返回列表第一个元素. */
    public XXX getFirst() {
        return first.item;
    }

    /** 在列表末尾插入 x. */
    public void addLast(XXX x) {
        size += 1;

        OneNode p = first;

        /* 把 p 当做指针顺藤摸瓜一直挪到列表末尾. */
        while (p.next != null) {
            p = p.next;
        }

        p.next = new OneNode(x, null);
    }

    /** 删除列表末尾的元素. */
    public void removeLast(){
        //自行补充...
    }

    public int size() {
        return size;
    }
}
```
可以看到，如果用户不小心把某节点x指回自己`x.next=x`,那就会进入死循环，所以我们需要把`OnoNode`实例`first`声明为私有变量已提供必要的保护。

### 超载
Overloading

如果想初始化一个空列表, 可以:
```java
/** 构造一个空列表. */
public LinkedList() {
    fist = null;
    size = 0;
}
```
即使原来已经有一个带参数x的构造器了, 这里再加一个同名构造器也没问题. 因为Java允许有不同参数的方法重名, 即超载 overloading.

### 程序不变条件
Invariants

上面超载了一个初始化空列表的构造器, 加入初始化一个空列表，然后直接调用`addLast`，程序会报错, 因为`null`没有`next`.

有几种修改方法, 比如用`if else`这种加特例的方法. 这个方案虽然可以能解决问题，但是应尽量避免加入特例代码。毕竟有特例就意味着增加了复杂度和额外的代码特例记忆需求, 而人记忆是有限的.

一个更简洁（尽管不太显而易见）的解决方案是修改数据结构本身，让所有`LinkedList`，维护起来都没有差别，即使是空的。如果把列表比做拉货的火车，那么货物就是列表承载的数据。一列火车如果只有车厢而没有车头（或者车尾）的话是没有意义的，因为没有动力。所以不管火车有没有拉货，有车厢还是没车厢，要称之为火车我们至少需要一个火车头，通过创建一个特殊节点 - 前哨节点 sentinel。前哨节点将保存一个值，具体数值我们不关心，它只是作为火车头，不装货。![](https://joshhug.gitbooks.io/hug61b/content/chap2/fig22/three_item_sentenlized_SLList.png "image from: https://joshhug.gitbooks.io/")
所以我们要修改`LinkedList`为：
```java
/* 第一个元素 （假如有的话）就是 sentinel.next. */
public class LinkedList<XXX> {
    private class OneNode {
        //...
    }

    private OneNode sentinel;
    private int size;

    /** 构造一个空列表. */
    public LinkedList() {
        sentinel = new OneNode(null, null);
        size = 0;
    }

    /** 构造一个初始元素为x的列表. */
    public LinkedList(XXX x) {
        sentinel = new OneNode(null, null);
        sentinel.next = new OneNode(x, null);
        size = 1;
    }
}
```
对于像`LinkedList`这样简单的数据结构来说，特例不多，我们也许可以hold住, 一旦后续遇到像树tree等更复杂的数据结构，控制特例数量就显得极为重要了。所以现在就要培养自己的这方面的习惯，保持程序不变条件成立。所谓 invariants 就是指数据结构任何情况下都是不会出错（除非程序有bug）.

具有前哨节点的`LinkedList`至少具有以下 invariants：
* 列表默认存在前哨节点。
* 列表第一个元素（如果非空的话）总是在`sentinel.next.item`。
* size变量始终是已添加的元素总数。

不变条件使得代码的推敲变得更加容易，同时给程序员提供了能够确保代码正常工作的具体目标。
