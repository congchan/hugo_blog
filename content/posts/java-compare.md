title: Java 比较对象大小
date: 2017-05-29
categories:
- CS
tags:
- Java
---
对物品排序首先需要比较各个物品的大小, 这个大小的定义既可以是按照"自然顺序", 也可以是其他指定的特殊规则.
<!-- more -->

### Comparable
Java的对象不能直接使用`>, <, =`进行比较. 在Python或C++中，当应用于不同对象类型时，比较运算符可以重新定义，但Java不支持。但可以借用接口继承，Java提供了一个`Comparable`接口，包含一个`compareTo`方法, 以保证任何实现该接口的类可以和其他同类做比较：
```java
/** Return negative if this < o.
    Return 0 if this equals o.
    Return positive if this > o.
*/
public interface Comparable<T> {
    public int compareTo(T obj);
}
```
当有class需要与其他class比较时, 就实现这个接口:
```java
public class Dog implements Comparable<Dog> {
    ...
    public int compareTo(Dog uddaDog) {
        return this.size - uddaDog.size;
    }
}
```
`Comparable`定义了类用于比较的自然顺序（Natural order）, 返回的是三种结果`负整数, 0, 正整数`, 分别对应小于, 等于和大于. `Insertion.sort()`不需要知道要排序的数组类型, 因为它直接调用数组成员自带的`compareTo`方法. Java的`Integer, Double, String, Date, File`数据类型都扩展了`Comparable`接口。
> A comparable object is capable of comparing itself with another object

### Comparator
如果我们想用灵活的不同方式对类进行比较比较呢？比如对音乐库里的歌曲根据艺术家、歌名等排序，二者都是`String`, 但一个类里面的`Comparable`只能有一个, 所以仅仅靠`Comparable`接口不够. 在Python可以使用HOF，编写新的比较函数，然后直接以参数形式传递该函数。

Java的方案是使用`Comparator`接口：
```java
public interface Comparator<T> {
    int compare(T o1, T o2);
}
```
比如Java系统自带的sort:
```java
/**
・Create Comparator object.
・Pass as second argument to Arrays.sort().
*/
String[] a;
...
Arrays.sort(a); // uses natural order
...
// uses alternate order defined by Comparator<String> object
Arrays.sort(a, String.CASE_INSENSITIVE_ORDER);
```
`insertion sort`的解决思路类似:
・Use Object instead of `Comparable`.
・Pass `Comparator` to `sort()` and `less()`.
```java
public static void sort(Object[] a, Comparator comparator) {
    int N = a.length;
    for (int i = 0; i < N; i++)
        for (int j = i; j > 0 && less(comparator, a[j], a[j-1]); j--)
            exch(a, j, j-1);
}

private static boolean less(Comparator c, Object v, Object w) {
    return c.compare(v, w) < 0;
}

private static void exch(Object[] a, int i, int j) {
    Object swap = a[i]; a[i] = a[j]; a[j] = swap;
}
```
需要自定义时, 根据需要在class内部编写实现`Comparator`接口的(嵌套)类, 并实现`compare`方法:
```java
public class Student {
    public static final Comparator<Student> BY_NAME = new ByName();
    public static final Comparator<Student> BY_SECTION = new BySection();
    private final String name;
    private final int section;
    ...
    private static class ByName implements Comparator<Student> {
        // 直接利用 String 已经定义好的 compareTo
        public int compare(Student v, Student w) {
            return v.name.compareTo(w.name);
        }
    }

    private static class BySection implements Comparator<Student> {
        public int compare(Student v, Student w) {
            return v.section - w.section;
        }
    }
}
```
在其他函数中调用时
```java
Student s1;
Student s2;
...
if (Student.BY_NAME.compare(s1, s2) > 0) {
    ...
}

...
Arrays.sort(a, Student.BY_NAME);
Arrays.sort(a, Student.BY_SECTION);
```
同理若需要增加其他判断标准，就创建新的实现`Comparator`的 class.

`Comparator`是可以将两个对象进行比较的第三方对象。由于只有一个`compareTo`的空间，如果想要支持不同方式进行比较，则要使用不同的`Comparator`。
> A Comparator is its own definition of how to compare two objects, and can be used to compare objects in a way that might not align with the natural ordering.
