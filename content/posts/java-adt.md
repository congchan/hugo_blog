title: Java 抽象数据类型
date: 2017-05-29
categories:
- CS
tags:
- Java
---
## Abstract Data Types (ADTS)
ADTS 是由其行为属性定义的抽象类型, 跟如何实现无关.

堆栈 Stacks 和队列 Queues 是两种类似的线性集合。堆栈是后进先出的ADT：元素总是从数据结构的一端添加或删除。队列是先进先出的ADT. 二者都支持以下操作:
`push()`: 加入
`peek()`: 返回下一个
`poll()`: 返回下一个并删除

Java的`Deque`(double ended queue, “deck”) 接口融合了堆栈和队列, 支持两端的元素插入和移除. `ArrayDeque`和`LinkedListDeque`都是实现`deque`这个接口，`deque`只是罗列了一些 methods，也即是一种合约，保证会实现的行为。而这些方法的具体实现则是由`ArrayDeque`和`LinkedListDeque`完成。从概念上讲，`deque`就是一种抽象的数据类型，只说会有什么行为，但不体现这些行为的具体实现方式，所以是抽象的。

优先级队列 priority queue 的每个元素都有一个与之关联的优先级，以决定从队列中元素操作的顺序。
<!-- more -->

三种方式实现`Stack`的`push(Item x)`:
1. 继承模式：`extends LinkedList<Item>`以使用其方法:
```java
public class ExtensionStack<Item> extends LinkedList<Item> {
    public void push(Item x) {
        add(x);
    }
}
```
2. 委托模式**Delegation**， 生成`Linked List`并调用其方法来达到目的
```java
public class DelegationStack<Item> {
    private LinkedList<Item> L = new LinkedList<Item>();
    public void push(Item x) {
        L.add(x);
    }
}
```
3. 类似委托模式, 只是这里可以利用任何实现了`List`接口的类, 如`Linked List, ArrayList`, 等等
```java
public class StackAdapter<Item> {
    private List L;
    public StackAdapter(List<Item> worker) {
        L = worker;
    }

    public void push(Item x) {
        L.add(x);
    }
}
```
Delegation vs Extension: Extension 一般是基于对父类有比较清楚的了解认知下才会使用。此外，扩展基本上等于在说明正在扩展的类与被扩展类是相似的。如果两个类无法看做是同属的, 那么就用委托模式。

Views: 通过视图进行的更改会影响底层对象。
```java
/** Create an ArrayList. */
List<String> L = new ArrayList<>();
/** Add some items. */
L.add(“at”); L.add(“ax”); …
List<String> SL = l.subList(1, 4);
/** Mutate that thing. */
SL.set(0, “jug”);
```

### API's
> An API(Application Programming Interface) of an ADT is the list of constructors and methods and a short description of each.

API 包括语法规范和语义规范
- 编译器确认语法符合要求
- 测试以帮助确认语义描述是否正确

### Java Libraries
Java有一些内置的抽象数据类型，打包在Java库中。 三个最重要的ADTs来自`java.util`库：
* List 列表：一个有序的元素集合，如`ArrayList`
* Set 集合：元素严格唯一（不重复）的(无序)集合，如`HashSet`
* Map 映射：A collection of `Key - value` 映射, `key`是唯一的。通过`key`访问`value`，如`HashMap`。

```java
/** takes in a String inputFileName
and puts every word from the input file into a list*/
public static List<String> getWords(String inputFileName) {
    List<String> lst = new ArrayList<String>();
    In in = new In();
    while (!in.isEmpty()) {
        lst.add(in.readString());
    }
    return lst;
}

/** takes in a List<String> and counts how many unique words there are in the file.*/
public static int countUniqueWords(List<String> words) {
    Set<String> ss = new HashSet<>();
    for (String s : words) {
           ss.add(s);
    }
    return ss.size();
}

/** takes in a List<String> targets and a List<String> words,
and finds the number of times each target word appears in the word list.*/
public static Map<String, Integer> collectWordCount(List<String> words) {
    Map<String, Integer> counts = new HashMap<String, Integer>();
    for (String t: target) {
        counts.put(s, 0);
    }
    for (String s: words) {
        if (counts.containsKey(s)) {
            counts.put(word, counts.get(s)+1);
        }
    }
    return counts;
}
```
![](https://joshhug.gitbooks.io/hug61b/content/assets/collection_hierarchy.png "Collections represent a group of objects, known as its elements: the white boxes are interfaces. The blue boxes are concrete classes. image from:https://joshhug.gitbooks.io/hug61b/")
通过设置环境变量（如`CLASSPATH = `）让Java编译器/解释器知道去哪里找 libraries。

`CLASSPATH`：Linux or MacOS, paths are separated by `:`. In Windows, paths are separated by `;`.
* `/home/--/--/javalib/*`, 在`.class`和`.jar`文件内查找依赖包，用于指定绝对路径。有同名时，会根据环境变量的先后顺序去排序靠前的。
* `./`指当前目录，`../`指上一层目录，用于指定相对路径。
* 也可以指定`classpath`, 这样系统的`CLASSPATH`会被忽略: `javac -cp ./:/home/stuff/:../ Foo.java`, 当有重名时, 选择顺序就是指明的路径顺序（当前目录-stuff目录-上一层目录）

IntelliJ会忽略`CLASSPATH`，它会自动调用`-cp`, 变量是基于当前项目指定的 libraries.
```java
/** 查看 IntelliJ 使用的 classpath*/
import java.net.URL;
import java.net.URLClassLoader;

public static void main(String[] args) {
    ClassLoader cl = ClassLoader.getSystemClassLoader();

    URL[] urls = ((URLClassLoader)cl).getURLs();

    for(URL url: urls){
        System.out.println(url.getFile());
    }
}
```

Build Systems：可以简单地将文件放入适当的位置，然后通过 Maven, Ant 和 Gradle 等工具使用 Build Systems 来自动设置项目, 省去了手动加载一长串 libraries.
