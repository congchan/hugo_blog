title: Java 09 | 数据结构 - 双向链表 Doubly Linked List
date: 2017-01-13
categories:
- CS
tags:
- Software Engineer
- Java
---
## 双向链表
Doubly Linked List

前面介绍过的单向链表有几个缺点. 第一个就是它的`addLast`操作非常慢。单向链表只有一个变量保存列表头的地址, 以及每个节点对后面节点的单向引用(链接). 对于很长的列表，`addLast`方法必须遍历整个列表, 直到找到列表末尾才能执行插入操作.
<!-- more -->
最直观的优化方案就是加个'车尾'![](/images/sllist_last_pointer.png "image from: https://joshhug.gitbooks.io/") 这样我们就可以直接通过`last.next`引用末尾位置.

不过另一个问题并没有解决, 就是删除列表最后一项`removeLast`这个操作还是很慢。因为在目前的结构设计下, 我们需要先找到倒数第二项，然后将其下一个指针设置为`null`。而要找到倒数第二节点, 我们就得先找到倒数第三个节点...... 以此类推。也就是说，对于删除末尾的操作，还是要几乎遍历整个列表。

### 反方向的链接
基于前面单向链表构建双向链表, 一个比较有效的方法是额外为每个节点添加一个指向前面节点的链接 - 指针.
```java
public class OneNode {
    public OneNode prev; //指向前
    public int item;
    public OneNode next; //指向后
}
```
增加这些额外的指针会导致额外的代码复杂度, 以及额外的内存开销, 这就是追求时间效率的代价.

### Sentinel 与尾节点
双向链表的一个设计初衷，就是为了解决单向链表针对列表末尾位置的操作效率不高的问题，除了sentinel和反方向的链接还不够，我们还需要一个节点（指针）能够直接帮我们定位到列表末端。可以考虑添加一个的尾节点`last`![](https://joshhug.gitbooks.io/hug61b/content/chap2/fig23/dllist_basic_size_0.png "image from: https://joshhug.gitbooks.io/") ![](https://joshhug.gitbooks.io/hug61b/content/chap2/fig23/dllist_basic_size_2.png "image from: https://joshhug.gitbooks.io/") 这样的列表就可以支持`O(1)`复杂度的`addLast`,`getLast` 和 `removeLast`操作了。

## 循环双端队列
Circular double ended queue

上面的尾节点设计虽然没什么错误，但有点瑕疵：最后一个尾节点指针有时指向前哨节点，有时指向一个真正的节点。更好的方法是使双向链表首尾相连, 构成一个循环，即前后节点共享唯一的一个前哨节点。![](https://joshhug.gitbooks.io/hug61b/content/chap2/fig23/dllist_circular_sentinel_size_0.png "image from: https://joshhug.gitbooks.io/") ![fig source https://joshhug.gitbooks.io/hug61b/content/chap2/fig23/dllist_circular_sentinel_size_2.png](https://joshhug.gitbooks.io/hug61b/content/chap2/fig23/dllist_circular_sentinel_size_2.png "image from: https://joshhug.gitbooks.io/")
这样的设计相对更整洁，更美观(主观上的), sentinel的`prev`就指向列表最后一个节点, sentinel的`next`指向列表第一个节点.
```java
public class LinkedListDeque<GType> {
    private class OneNode {
        public OneNode prev;
        public GType item;
        public OneNode next;

        public OneNode(OneNode p, GType i, OneNode n) {
            prev = p;
            item = i;
            next = n;
        }
    }
}
```
Sentinel's forward link always points to the last element.
Sentinel's backward link always points to the first element.

然后修改构造函数:
```java
/** Creates an empty deque. */
public LinkedListDeque(){
    sentinel = new OneNode(null,null, null);
    sentinel.prev = sentinel;
    sentinel.next = sentinel;
    size = 0;
}

/** Creates a deque with x  */
public LinkedListDeque(GType x){
    sentinel = new OneNode(null, null, null);
    sentinel.next = new OneNode(sentinel, x, sentinel);
    sentinel.prev = sentinel.next;
    size = 1;
}
```
如果初始化的是空列表, 其实就是一个自己指向自己的`sentinel`节点. 如果是非空列表, 那么`sentinel`节点和真实的节点就构成了一个最简单的二元循环体.

### 针对列表末尾位置的操作
双端链表结构优雅，虽然某些操作如`addFirst`等编码复杂度会提高, 但不影响速度. 更重要的是, 相比单向链表, 它反而使得`addLast, moveLast`等方法的代码实现变得简单了, 而且还进一步提升了运行速度(`从O(n)到O(c)`).
```java
/** Adds an item to the back of the Deque - O(c) */
public void addLast(GType x){
    OneNode oldBackNode = sentinel.prev;
    OneNode newNode = new OneNode(oldBackNode, x, sentinel);
    sentinel.prev = newNode;
    oldBackNode.next = newNode;
    size += 1;
}

/** Removes and returns the item at the front of the Deque.
 * If no such item exists, returns null.O(c). */
public GType removeFirst(){
    if (isEmpty()){
        return null;
    }

    OneNode oldFrontNode = sentinel.next;
    sentinel.next = oldFrontNode.next;
    oldFrontNode.next.prev = sentinel;
    size -= 1;
    return oldFrontNode.item;
}
```
