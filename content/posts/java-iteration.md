---
title: Java 迭代 Iteration
date: 2017-05-29
author: "Cong Chan"
tags: ['Java']
---
Java提供了 foreach (enhanced for) 的循环简写语法:
```java
ArrayMap<String, Integer> am = new ArrayMap<String, Integer>();

for (String s : am) {
    System.out.println(s);
}
```
实现的关键原理是使用`Iterable`接口使一个类变成可迭代的: 该接口包含一个`iterator()`方法用于返回一个`Iterator`对象。`Iterator`接口定义`Iterator`对象和`hasNext(), next()`方法来进行实际的迭代操作。
<!-- more -->
```java
public class ArrayMap<K, V> implements Map61B<K, V>, Iterable<K> {
    private K[] keys;
    private V[] values;
    int size;

    public ArrayMap() {
        keys = (K[]) new Object[100];
        values = (V[]) new Object[100];
        size = 0;
     }

    @Override
    public Iterator<T> iterator() {
        return new KeyIterator();
    }

    public class KeyIterator implements Iterator<K> {
        private int ptr;
        public KeyIterator() { ptr = 0; }
        public boolean hasNext() { return (ptr != size); }
        public K next() {
            K returnItem = keys[ptr];
            ptr = ptr + 1;
            return returnItem;
        }
    }
}
```
不同的数据结构，`Iterator`有不同的实现方式.

`KeyIterator`即使是`private`也可以编译, 因为`iterator()`在这里是`public`的:
```java
import java.util.Iterator;

public class Demo{
    public static void main(String[] args) {
        ArrayMap<String, Integer> am = new ArrayMap<String, Integer>();

        Iterator<String> it = am.iterator();

        for (String s : am) { ... }
    }1
}
```
除了用嵌套类来自定义实现`Iterator`, 也可以利用数据结构本身的特性. 比如`ArrayMap`里面刚好包含一个可迭代的数据结构`List keys`
```java
public Iterator<T> iterator() {
    List<K> keylist = keys();
    return keylist.Iterator();
}
```

**注意要点**
* `hasNext()`的判断依据是**当前状态下能返回至少一个成员**, 不要混淆为*下一次能否返回*: 因为迭代时过程中, 每次调用`next()`之前, java 都会先调用`hasNext()`.
* 实现方法时, 要保证第一次`next()`返回的是第一个成员.
