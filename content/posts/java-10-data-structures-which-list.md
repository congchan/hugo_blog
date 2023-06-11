---
title: Java 10 | 数据结构 - LinkedList 还是 ArrayList
date: 2017-01-28
author: "Cong Chan"
tags: ['Software Engineer', 'Java']
---
Java 提供了 `ArrayList`, `ArrayDeque` 和 `LinkedList` 几个API.

队列 queue, 通俗的含义, 就是不能插队, 只能在末尾插入.

> 双端队列 Double Ended Queue (Deque) 是具有动态大小的序列容器，可以在两端（前端或后端）扩展或收缩
--http://www.cplusplus.com/reference/deque/deque/

CS61b的[project 1a](http://sp18.datastructur.es/materials/proj/proj1a/proj1a)需要实现两种双端队列（array based 和 linkedklist based）.

不同的API, 在考虑什么时候应该用哪个时, 我们需要考虑它们的性能差异:
* **搜索/定位**：与`LinkedList`相比，`ArrayList`搜索更快。 `ArrayList`的`get(int index)`性能是`O(1)`的，而LinkedList的性能是`O(n)`。因为`ArrayList`基于`array`数据结构，可以直接用 array index 定位元素。
* **删除/插入**：`LinkedList` 操作性能是`O(1)`，而`ArrayList`的性能从`O(n)`（删除/插入第一个元素）到`O(n)`（最后一个元素）都有可能。因为`LinkedList`的每个元素都包含两个指向其相邻前后元素的指针（地址），因此仅需要改变，被删节点的`prev`和`next`指针位置。而在`ArrayList`中，需要移动剩余元素，来重新填充`array`空间。
* **内存开销**：`LinkedList`的每个元素都有更多的内存开销(额外的指针), 而`ArrayLists`没有这个开销。但是，`ArrayLists`需要占用初始容量。一般`ArrayList`的默认初始容量非常小（Java 1.4 - 1.8使用10）。但是，往`ArrayLists`添加元素时， 它可能会适当地增大容量，所以如果添加了很多元素，则必须不断调整数组的大小，那样也可能会导致元素频繁挪动位置。

综上所述：
1. 如果在应用中需要**频繁插入和删除**，那么选择`LinkedList`。
2. 假如一开始，就知道后面要添加大量元素，那就使用较高的初始容量来构造`ArrayList`。
3. 大部分用例中, 相比LinkedList, 人们更偏爱ArrayList以及ArrayDeque。如果你不确定应该选哪个, 那么就直接考虑ArrayList吧(参考 https://stackoverflow.com/questions/322715/when-to-use-linkedlist-over-arraylist).
