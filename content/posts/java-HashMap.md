---
title: Java HashMap
date: 2017-06-19
author: "Cong Chan"
tags: ['Java']
---
## HashMap
HashMap 是基于哈希表的 Map 接口的非同步实现。此实现提供所有可选的映射操作，并允许使用 null 值和 null 键。此类不保证映射的顺序，特别是它不保证该顺序不是stable的。
<!-- more -->
HashMap 底层就是一个数组结构，数组中的每一项又是一个链表。![](/images/hash_map.png "image from: https://algs4.cs.princeton.edu/")
当新建一个 HashMap 的时候，就会初始化一个数组`table = new Entry[capacity];`, 每个 `Entry` 是一个 key-value 对，有一个指向下一个元素的引用，这就构成了链表。
```java
public HashMap(int initialCapacity, float loadFactor) {
    if (initialCapacity < 0)
        throw new IllegalArgumentException("Illegal initial capacity: " +
                                           initialCapacity);
    if (initialCapacity > MAXIMUM_CAPACITY)
        initialCapacity = MAXIMUM_CAPACITY;
    if (loadFactor <= 0 || Float.isNaN(loadFactor))
        throw new IllegalArgumentException("Illegal load factor: " +
                                           loadFactor);

    // Find a power of 2 >= initialCapacity
    int capacity = 1;
    while (capacity < initialCapacity)
        capacity <<= 1;

    this.loadFactor = loadFactor;
    threshold = (int)Math.min(capacity * loadFactor, MAXIMUM_CAPACITY + 1);
    table = new Entry[capacity];
    useAltHashing = sun.misc.VM.isBooted() &&
            (capacity >= Holder.ALTERNATIVE_HASHING_THRESHOLD);
    init();
}

static class Entry<K,V> implements Map.Entry<K,V> {
    final K key;
    V value;
    Entry<K,V> next;
    final int hash;
    ……
}
```
HashMap 底层数组的长度总是 2 的 n 次方，这是 HashMap 在速度上的优化, 所以有这段代码保证初始化时 HashMap 的容量总是 2 的 n 次方，即底层数组的长度总是为 2 的 n 次方。
```java
// Find a power of 2 >= initialCapacity
int capacity = 1;
while (capacity < initialCapacity)
    capacity <<= 1;
```

### 插入/更新
如果 key 存在了，那么新的 value 会代替旧的 value，并且如果 key 存在的情况下，该方法返回的是旧的 value，如果 key 不存在，那么返回 null。
```java
/**
 * Associates the specified value with the specified key in this map.
 * If the map previously contained a mapping for the key, the old
 * value is replaced.
 *
 * @param key key with which the specified value is to be associated
 * @param value value to be associated with the specified key
 * @return the previous value associated with <tt>key</tt>, or
 *         <tt>null</tt> if there was no mapping for <tt>key</tt>.
 *         (A <tt>null</tt> return can also indicate that the map
 *         previously associated <tt>null</tt> with <tt>key</tt>.)
 */
public V put(K key, V value) {
    //其允许存放null的key和null的value，当其key为null时，调用putForNullKey方法，放入到table[0]的这个位置
    if (key == null)
        return putForNullKey(value);
    //通过调用hash方法对key进行哈希，得到哈希之后的数值。该方法实现可以通过看源码，其目的是为了尽可能的让键值对可以分不到不同的桶中
    int hash = hash(key);
    //根据上一步骤中求出的hash得到在数组中是索引i
    int i = indexFor(hash, table.length);
    //如果i处的Entry不为null，则通过其next指针不断遍历e元素的下一个元素。
    for (Entry<K,V> e = table[i]; e != null; e = e.next) {
        Object k;
        if (e.hash == hash && ((k = e.key) == key || key.equals(k))) {
            V oldValue = e.value;
            e.value = value;
            e.recordAccess(this);
            return oldValue;
        }
    }

    modCount++;
    addEntry(hash, key, value, i);
    return null;
}
```
往 HashMap 中 put 元素的时候，先根据 key 的 hashCode 重新计算 hash 值，根据 hash 值得到这个元素在数组中的位置（即下标），如果数组该位置上已经存放有其他元素了，那么在这个位置上的元素将以链表的形式存放，新加入的放在链头，最先加入的放在链尾。如果数组该位置上没有元素，就直接将该元素放到此数组中的该位置上。

`addEntry(hash, key, value, i)`方法根据计算出的 hash 值，将 key-value 对放在数组 table 的 i 索引处。首先要判断容量如果超过阈值`threshold`, 并且索引位置不为空, 就要先扩容：
```java
/**
 * Adds a new entry with the specified key, value and hash code to
 * the specified bucket.  It is the responsibility of this
 * method to resize the table if appropriate.
 *
 * Subclass overrides this to alter the behavior of put method.
 */
void addEntry(int hash, K key, V value, int bucketIndex) {
    if ((size >= threshold) && (null != table[bucketIndex])) {
        resize(2 * table.length);
        hash = (null != key) ? hash(key) : 0;
        bucketIndex = indexFor(hash, table.length);
    }

    createEntry(hash, key, value, bucketIndex);
}
void createEntry(int hash, K key, V value, int bucketIndex) {
    // 获取指定 bucketIndex 索引处的 Entry
    Entry<K,V> e = table[bucketIndex];
    // 将新创建的 Entry 放入 bucketIndex 索引处，并让新的 Entry 指向原来的 Entry e
    table[bucketIndex] = new Entry<>(hash, key, value, e);
    size++;
}
```
hash(int h)方法根据 key 的 hashCode 重新计算一次散列。此算法加入了高位计算，防止低位不变，高位变化时，造成的 hash 冲突。
```java
final int hash(Object k) {
    int h = 0;
    if (useAltHashing) {
        if (k instanceof String) {
            return sun.misc.Hashing.stringHash32((String) k);
        }
        h = hashSeed;
    }
    //得到k的hashcode值
    h ^= k.hashCode();
    //进行计算
    h ^= (h >>> 20) ^ (h >>> 12);
    return h ^ (h >>> 7) ^ (h >>> 4);
}
```

### 高效计算索引
我们希望 HashMap 里面元素位置尽量的分布均匀，尽量使得每个位置上的元素数量只有一个，那么当用 hash 算法求得这个位置的时候，马上就可以知道对应位置的元素，而不用再去遍历链表，这样就大大优化了查询的效率。

对于任意给定的对象，只要它的 `hashCode()` 返回值相同，那么程序调用 `hash(int h)` 方法所计算得到的 hash 码值总是相同的。我们首先想到的就是把 hash 值对数组长度取模运算，这样一来，元素的分布相对来说是比较均匀的。但是，“模”运算的消耗还是比较大的，在 HashMap 中是这样做的：调用 `indexFor(int h, int length)` 方法来计算该对象应该保存在 table 数组的哪个索引处。
```java
/**
     * Returns index for hash code h.
     */
static int indexFor(int h, int length) {
    return h & (length-1);
}
```
当 length 总是 2 的 n 次方时，`h & (length-1)`运算等价于对 length **取模**，也就是 `h % length`，但是位运算比 `%` 更高效。
```
             hash 	table.length-1
8 & (15-1)： 0100 & 	1110 	      = 0100
9 & (15-1)： 0101 & 	1110 	      = 0100
8 & (16-1)： 0100 & 	1111 	      = 0100
9 & (16-1)： 0101 & 	1111 	      = 0101
```
15不是以2为底的真数, `15-1 = 0b1110` 最后一位永远是 0, 任何与之`&`的hash值最后一位也永远是0, 这就导致 `0001，0011，0101，1001，1011，0111，1101` 这几个位置永远都没机会存放元素了，空间浪费相当大，更糟的是这种情况中，数组可以使用的位置比数组长度小了很多，这意味着进一步增加了碰撞的几率，减慢了查询的效率！

而如果`length`是以2底的真数, 其二进制高位一定是`1`, 且只有这一个`1`, 如`0b1000`, 那么`length-1`就会变成只有该位置是`0`其右边低位所有位变为`1`, 即`0b0111`. 这使得在低位上与操作时，得到的和原 hash 的低位相同. 加之 `hash(int h)`方法对 key 的 hashCode 的进一步优化，加入了高位计算，就使得只有完全相同的 hash 值的两个key才会被放到数组中的同一个位置上形成链表。所以如果数组长度为 2 的 n 次幂的时候，不同的 key 算得得 index 相同的几率较小，那么数据在数组上分布就比较均匀，也就是说碰撞的几率小，相对的，查询的时候就不用遍历某个位置上的链表，这样查询效率也就较高了。

另一种计算Hash的方法: 通过hashCode()的高16位异或低16位实现的：`(h = k.hashCode()) ^ (h >>> 16)`，主要是从速度、功效、质量来考虑的，这么做可以在数组table的length比较小的时候，也能保证考虑到高低Bit都参与到Hash的计算中，同时不会有太大的开销。
```java
//jdk1.8 & jdk1.7
static final int hash(Object key) {
     int h;
     // h = key.hashCode() 为第一步 取hashCode值
     // h ^ (h >>> 16)  为第二步 高位参与运算
     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
}
```
一旦出现链表过长，严重影响HashMap的性能，所以JDK1.8版本对数据结构做了进一步的优化，引入了红黑树。当链表长度太长（默认超过8）时，链表就转换为红黑树，利用红黑树快速增删改查的特点提高HashMap的性能，其中会用到红黑树的插入、删除、查找等算法

### 读取
从 HashMap 中 get 元素时，首先计算 key 的 hashCode，找到数组中对应位置的某一元素，然后通过 key 的 equals 方法在对应位置的链表中找到需要的元素。
```java
/**
 * Returns the value to which the specified key is mapped,
 * or {@code null} if this map contains no mapping for the key.
 *
 * <p>More formally, if this map contains a mapping from a key
 * {@code k} to a value {@code v} such that {@code (key==null ? k==null :
 * key.equals(k))}, then this method returns {@code v}; otherwise
 * it returns {@code null}.  (There can be at most one such mapping.)
 *
 * <p>A return value of {@code null} does not <i>necessarily</i>
 * indicate that the map contains no mapping for the key; it's also
 * possible that the map explicitly maps the key to {@code null}.
 * The {@link #containsKey containsKey} operation may be used to
 * distinguish these two cases.
 *
 * @see #put(Object, Object)
 */
public V get(Object key) {
    if (key == null)
        return getForNullKey();
    Entry<K,V> entry = getEntry(key);

    return null == entry ? null : entry.getValue();
}

final Entry<K,V> getEntry(Object key) {
    int hash = (key == null) ? 0 : hash(key);
    for (Entry<K,V> e = table[indexFor(hash, table.length)];
         e != null;
         e = e.next)
    {
        Object k;
        if (e.hash == hash &&
            ((k = e.key) == key || (key != null && key.equals(k))))
            return e;
    }
    return null;
}
```

### resize
rehash
当 HashMap 中的元素越来越多的时候，hash 冲突的几率也就越来越高，因为数组的长度是固定的。所以为了提高查询的效率，就要对 HashMap 的数组进行扩容。

HashMap 数组扩容之后，最消耗性能的点就出现了：原数组中的数据必须重新计算其在新数组中的位置，并放进去，这就是 resize。

当 HashMap 中的元素个数超过`数组大小 * loadFactor`时，就会进行数组扩容，loadFactor的默认值为 `0.75`，这是一个折中的取值。也就是说，默认情况下，数组大小为 16，那么当 HashMap 中元素个数超过 `16*0.75=12` 的时候，就把数组的大小扩大一倍, 扩展为 `2*16=32`，然后重新计算每个元素在数组中的位置. 所以如果我们已经预知 HashMap 中元素的个数，那么预设元素的个数能够有效的提高 HashMap 的性能。

JDK1.7的HashMap在实现`resize()`时，新`table[]`的列表采用LIFO方式，即队头插入。这样做的目的是**避免尾部遍历**。
```java
void resize(int newCapacity) {   //传入新的容量
     Entry[] oldTable = table;    //引用扩容前的Entry数组
     int oldCapacity = oldTable.length;
     if (oldCapacity == MAXIMUM_CAPACITY) {  //扩容前的数组大小如果已经达到最大(2^30)了
          threshold = Integer.MAX_VALUE; //修改阈值为int的最大值(2^31-1)，这样以后就不会扩容了
         return;
     }

     Entry[] newTable = new Entry[newCapacity];  //初始化一个新的Entry数组
     transfer(newTable);                         //！！将数据转移到新的Entry数组里
     table = newTable;                           //HashMap的table属性引用新的Entry数组
   threshold = (int)(newCapacity * loadFactor);//修改阈值
}

// 原Entry数组的元素拷贝到新的Entry数组里
void transfer(Entry[] newTable) {
      Entry[] src = table;                   //src引用了旧的Entry数组
      int newCapacity = newTable.length;
      for (int j = 0; j < src.length; j++) { //遍历旧的Entry数组
          Entry<K,V> e = src[j];             //取得旧Entry数组的每个元素
          if (e != null) {
              src[j] = null;//释放旧Entry数组的对象引用（for循环后，旧的Entry数组不再引用任何对象）
              do {
                  Entry<K,V> next = e.next;
                 int i = indexFor(e.hash, newCapacity); //！！重新计算每个元素在数组中的位置
                 e.next = newTable[i]; //标记[1], 变为头节点
                 newTable[i] = e;      //将元素放在数组上
                 e = next;             //访问下一个Entry链上的元素
             } while (e != null);
         }
     }
 }
```
`newTable[i]`的引用赋给了`e.next`，也就是使用了单链表的头插入方式，同一位置上新元素总会被放在链表的头部位置；这样先放在一个索引上的元素终会被放到Entry链的尾部(如果发生了hash冲突的话）

JDK1.8优化了重新计算hash这一步。因为使用的是2次幂的扩展(指长度扩为原来2倍)，所以，元素的位置要么是在原位置，要么是在原位置再移动2次幂的位置。元素在重新计算hash之后，因为n变为2倍，那么n-1的mask范围在高位多1位. 因此，在扩容HashMap的时候，不需要像JDK1.7的实现那样重新计算hash，只需要看原来的hash值在新增的那个bit位是1还是0就好了，是0的话索引没变，是1的话索引变成`原索引+oldCap`. 这样省去了重新计算hash值的时间，而且由于新增的1bit是0还是1可以认为是随机的，因此resize的过程，均匀的把之前的冲突的节点分散到新的bucket了。
```java
// JDK1.8
final Node<K,V>[] resize() {
    Node<K,V>[] oldTab = table;
    int oldCap = (oldTab == null) ? 0 : oldTab.length;
    int oldThr = threshold;
    int newCap, newThr = 0;
    if (oldCap > 0) {
        // 超过最大值就不再扩充了
        if (oldCap >= MAXIMUM_CAPACITY) {
            threshold = Integer.MAX_VALUE;
            return oldTab;
        }
        // 没超过最大值，就扩充为原来的2倍
        else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                 oldCap >= DEFAULT_INITIAL_CAPACITY)
            newThr = oldThr << 1; // double threshold
    }
    else if (oldThr > 0) // initial capacity was placed in threshold
        newCap = oldThr;
    else {               // zero initial threshold signifies using defaults
        newCap = DEFAULT_INITIAL_CAPACITY;
        newThr = (int)(DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
    }
    // 计算新的resize上限
    if (newThr == 0) {

        float ft = (float)newCap * loadFactor;
        newThr = (newCap < MAXIMUM_CAPACITY && ft < (float)MAXIMUM_CAPACITY ?
                  (int)ft : Integer.MAX_VALUE);
    }
    threshold = newThr;
    @SuppressWarnings({"rawtypes","unchecked"})
        Node<K,V>[] newTab = (Node<K,V>[])new Node[newCap];
    table = newTab;
    if (oldTab != null) {
        // 把每个bucket都移动到新的buckets中
        for (int j = 0; j < oldCap; ++j) {
            Node<K,V> e;
            if ((e = oldTab[j]) != null) {
                oldTab[j] = null;
                if (e.next == null)
                    newTab[e.hash & (newCap - 1)] = e;
                else if (e instanceof TreeNode)
                    ((TreeNode<K,V>)e).split(this, newTab, j, oldCap);
                else { // preserve order
                    Node<K,V> loHead = null, loTail = null;
                    Node<K,V> hiHead = null, hiTail = null;
                    Node<K,V> next;
                    do {
                        next = e.next;
                        // 原索引  oldCap - 1:   0 1111     oldCap : 1 0000       判断 key的 hash值的那一位是否为1分为两类
                        if ((e.hash & oldCap) == 0) {
                            if (loTail == null)
                                loHead = e;
                            else
                                loTail.next = e;
                            loTail = e;
                        }
                        // 原索引+oldCap
                        else {
                            if (hiTail == null)
                                hiHead = e;
                            else
                                hiTail.next = e;
                            hiTail = e;
                        }
                    } while ((e = next) != null);
                    // 原索引放到bucket里
                    if (loTail != null) {
                        loTail.next = null;
                        newTab[j] = loHead;
                    }
                    // 原索引+oldCap放到bucket里
                    if (hiTail != null) {
                        hiTail.next = null;
                        newTab[j + oldCap] = hiHead;
                    }
                }
            }
        }
    }
    return newTab;
}
```

### 扩容的线程不安全
扩容时存在条件竞争，如果两个线程都发现HashMap需要调整大小，它们会同时尝试调整大小。在调整的过程中，为了避免尾部遍历(tail traversing)而采用队头插入的方式，会让原先的链表顺序会反转。如果在多线程环境中发生条件竞争，会导致死循环。因此在并发环境下，使用CurrentHashMap来替代HashMap
```java
void transfer(Entry[] newTable) {
      ...
      do {
          Entry<K,V> next = e.next;
          int i = indexFor(e.hash, newCapacity); //！！重新计算每个元素在数组中的位置
          e.next = newTable[i]; //标记[1], 变为头节点
          newTable[i] = e;      //将元素放在数组上
          e = next;             //访问下一个Entry链上的元素
      } while (e != null);
 }
```
如果当 thread1 运行到 `int i = indexFor(e.hash, newCapacity);`时`e = key(3), next = key(7)`, 而thread2 已经执行完毕`transfer()`, 此时的状态是![](/images/HashMap_dead_lock01.png).
此时实际上已经完成了`transfer()`了, `key(3)`和`key(7)`顺序反转了, `key(7).next = key(3)`. 但thread1还没跑完, 此时Thread1的 `e = key(3)`，而`next = key(7)`，现在thread1继续运行,
1, `e.next = newTable[i];` 此时就是把`key(3).next`指向`key(7)`, 环形链接出现
2,  `newTalbe[i] = e;`, 把`key(3)`置于表头
![](/images/HashMap_dead_lock02.png).

JDK1.8的优化，通过增加tail指针，既避免了死循环问题（让数据直接插入到队尾），又避免了尾部遍历。
```java
// http://hg.openjdk.java.net/jdk8/jdk8/jdk/file/f4129fcfacdc/src/share/classes/java/util/HashMap.java
final Node<K,V>[] resize() {
    Node<K,V>[] oldTab = table;
    int oldCap = (oldTab == null) ? 0 : oldTab.length;
    int oldThr = threshold;
    int newCap, newThr = 0;
    if (oldCap > 0) {
        // 超过最大值就不再扩充了，就只好随你碰撞去吧
        if (oldCap >= MAXIMUM_CAPACITY) {
            threshold = Integer.MAX_VALUE;
            return oldTab;
        }
        // 没超过最大值，就扩充为原来的2倍
        else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                 oldCap >= DEFAULT_INITIAL_CAPACITY)
            newThr = oldThr << 1; // double threshold
    }
    else if (oldThr > 0) // initial capacity was placed in threshold
        newCap = oldThr;
    else {               // zero initial threshold signifies using defaults
        newCap = DEFAULT_INITIAL_CAPACITY;
        newThr = (int)(DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
    }
    // 计算新的resize上限
    if (newThr == 0) {
        float ft = (float)newCap * loadFactor;
        newThr = (newCap < MAXIMUM_CAPACITY && ft < (float)MAXIMUM_CAPACITY ?
                  (int)ft : Integer.MAX_VALUE);
    }
    threshold = newThr;
    @SuppressWarnings({"rawtypes","unchecked"})
        Node<K,V>[] newTab = (Node<K,V>[])new Node[newCap];
    table = newTab;
    if (oldTab != null) {
        // 把每个bucket都移动到新的buckets中
        for (int j = 0; j < oldCap; ++j) {
            Node<K,V> e;
            if ((e = oldTab[j]) != null) {
                oldTab[j] = null;
                if (e.next == null)
                    newTab[e.hash & (newCap - 1)] = e;
                else if (e instanceof TreeNode)
                    ((TreeNode<K,V>)e).split(this, newTab, j, oldCap);
                else { // preserve order
                    // // 新扩容部分，标识为hi，原来old的部分标识为lo
                    Node<K,V> loHead = null, loTail = null;
                    Node<K,V> hiHead = null, hiTail = null;
                    Node<K,V> next;
                    do {
                        next = e.next;
                        // 原索引
                        if ((e.hash & oldCap) == 0) {
                            if (loTail == null)
                                loHead = e;
                            else
                                loTail.next = e;
                            loTail = e;
                        }
                        // 原索引+oldCap
                        else {
                            if (hiTail == null)
                                hiHead = e;
                            else
                                hiTail.next = e;
                            hiTail = e;
                        }
                    } while ((e = next) != null);
                    // 原索引放到bucket里
                    if (loTail != null) {
                        loTail.next = null;
                        newTab[j] = loHead;
                    }
                    // 原索引+oldCap放到bucket里
                    if (hiTail != null) {
                        hiTail.next = null;
                        newTab[j + oldCap] = hiHead;
                    }
                }
            }
        }
    }
    return newTab;
}
```

### HashMap性能参数
HashMap 包含如下几个构造函数：
* HashMap()：构建一个初始容量为 16，负载因子为 0.75 的 HashMap。
* HashMap(int initialCapacity)：构建一个初始容量为 initialCapacity，负载因子为 0.75 的 HashMap。
* HashMap(int initialCapacity, float loadFactor)：以指定初始容量、指定的负载因子创建一个 HashMap。

负载因子 loadFactor 衡量的是一个散列表的空间的使用程度，负载因子越大表示散列表的装填程度越高，反之愈小。对于使用链表法的数组来说，查找一个元素的平均时间是 O(1+a)，因此如果负载因子越大，对空间的利用更充分，然而后果是查找效率的降低；如果负载因子太小，那么数组的数据将过于稀疏，对空间造成严重浪费。

HashMap 的实现中，通过 threshold 字段来判断 HashMap 的最大容量：`threshold = (int) (capacity * loadFactor);`

threshold 就是在给定 loadFactor 和 capacity 下允许的最大元素数目，超过这个数目就重新 resize，以降低实际的负载因子。默认的的负载因子 0.75 是对空间和时间效率的一个平衡选择。当容量超出此最大容量时， resize 后的 HashMap 容量是翻倍.

### 迭代中的线程不安全
如果在使用迭代器的过程中有其他线程修改了 HashMap，那么将抛出`ConcurrentModificationException`，这就是所谓 fail-fast 策略。Fail-fast 机制是 java 集合(Collection)中的一种错误机制。当多个线程对同一个集合的内容进行操作时，就可能会产生 fail-fast 事件。

这一策略的实现是通过 modCount变量，对 HashMap 内容（当然不仅仅是 HashMap 才会有，其他例如 ArrayList 也会）的修改都将增加这个值（在很多操作中都有 modCount++ 这句），那么在迭代器初始化过程中会将这个值赋给迭代器的 expectedModCount。
```java
HashIterator() {
    expectedModCount = modCount;
    if (size > 0) { // advance to first entry
    Entry[] t = table;
    while (index < t.length && (next = t[index++]) == null)
        ;
    }
}
```
在迭代过程中，判断 modCount 跟 expectedModCount 是否相等，如果不相等就表示已经有其他线程修改了 Map：modCount 声明为 `volatile`，保证线程之间修改的可见性。
```java
final Entry<K,V> nextEntry() {
    if (modCount != expectedModCount)
        throw new ConcurrentModificationException();
}
```
fail-fast 机制是一种错误检测机制, 它只能被用来检测错误，因为 JDK 并不保证 fail-fast 机制一定会发生。若在多线程环境下使用 fail-fast 机制的集合，建议使用`java.util.concurrent`包下的类。

HashMap 的遍历方式: 使用`entrySet()`的遍历效率较高
```java
Map map = new HashMap();
　　Iterator iter = map.entrySet().iterator();
　　while (iter.hasNext()) {
    　　Map.Entry entry = (Map.Entry) iter.next();
    　　Object key = entry.getKey();
    　　Object val = entry.getValue();
    }
```

### 什么wrapper类适合作为键
String, Interger.
因为不可变性质, String是不可变的，也是final的，而且已经重写了equals()和hashCode()方法了。其他的wrapper类也有这个特点。

不可变性是必要的，因为为了要计算hashCode()，就要防止键值改变，如果键值在放入时和获取时返回不同的hashcode的话，那么就不能从HashMap中找到你想要的对象。不可变性还有其他的优点如线程安全。如果可以仅仅通过将某个field声明成final就能保证hashCode是不变的，那么请这么做吧。因为获取对象的时候要用到equals()和hashCode()方法，那么键对象正确的重写这两个方法是非常重要的。如果两个不相等的对象返回不同的hashcode的话，那么碰撞的几率就会小些，这样就能提高HashMap的性能

可以使用自定义的对象作为键，只要它遵守了equals()和hashCode()方法的定义规则，并且当对象插入到Map中之后将不会再改变了。如果这个自定义对象时不可变的，那么它已经满足了作为键的条件，因为当它创建之后就已经不能改变了。

### HashMap和HashTable的比较
和Hash Table的假设一样，假定哈希函数将元素适当地分布在各区之间，可为基本操作（get 和 put）提供稳定的性能。

跟 Hash Table 相比， Hash Map的作者多了 [Doug Lea](http://g.oswego.edu/)。他写了`util.concurrent`包。著有并发编程`Concurrent Programming in Java: Design Principles and Patterns` 一书。

Hashtable是java一开始发布时就提供的键值映射的数据结构，但是现在Hashtable基本上已经被弃用了。而产生于JDK1.2的HashMap已经成为应用最为广泛的一种数据类型了。造成这样的原因是因为Hashtable是线程安全的（同步的），效率比较低。

#### 继承的父类不一样
HashMap是继承自`AbstractMap`类，而`HashTable`是继承自`Dictionary`类(Dictionary类已经被废弃)。不过它们都实现了同时实现了`map`、`Cloneable`、`Serializable`这三个接口

#### 对外提供的接口不同
HashMap比Hashtable少了`elments()`和`contains()`两个方法。HashMap 有`containsKEY()`和`containsValue()`, 事实上，`contansValue()`就是调用了`contains()`方法。

#### 对`Null`的支持不同
Hashtable既不支持Null key也不支持Null value。HashMap中，`null`可以作为键，这样的键只有一个；值可以为`null`且不做数量限制。当`get()`方法返回`null`时，可能是 HashMap 中没有该键，也可能使该键所对应的值为`null`。因此，在HashMap中不能由get()方法来判断HashMap中是否存在某个键，而应该用`containsKey()`方法来判断。

#### 线程安全性不同
Hashtable是线程安全的，它的每个方法中都加入了`Synchronize`方法。在多线程并发的环境下，可以直接使用Hashtable，不需要自己为它的方法实现同步. Hashmap 不是同步的(不是线程安全)，如果多个线程同时访问一个 HashMap，而其中至少一个线程从结构上（指添加或者删除一个或多个映射关系的任何操作）修改了，则必须保持外部同步，以防止对映射进行意外的非同步访问。

由于Hashtable是线程安全的也是synchronized，所以在单线程环境下它比HashMap要慢。如果不需要同步，只需要单一线程，那么使用HashMap性能要好过Hashtable。

#### 初始容量和扩容不同
Hashtable默认的初始大小为11，之后每次扩充，容量变为原来的`2n+1`。HashMap默认的初始化大小为16。之后每次扩充，容量变为原来的`2n`。

创建时，如果给定了容量初始值，那么Hashtable会直接使用你给定的大小，而HashMap会将其扩充为2的幂次方大小。也就是说Hashtable会尽量使用素数、奇数。而HashMap则总是使用2的幂作为哈希表的大小。

之所以会有这样的不同，是因为Hashtable和HashMap设计时的侧重点不同。Hashtable的侧重点是哈希的结果更加均匀，使得哈希冲突减少。当哈希表的大小为素数时，简单的取模哈希的结果会更加均匀。而HashMap则更加关注hash的计算效率问题。在取模计算时，如果模数是2的幂，那么我们可以直接使用**位运算**来得到结果，效率要大大高于做除法。HashMap为了加快hash的速度，将哈希表的大小固定为了2的幂。当然这引入了哈希分布不均匀的问题，所以HashMap为解决这问题，又对hash算法做了一些改动。这从而导致了Hashtable和HashMap的计算hash值的方法不同

#### 计算hash的方法不同
Hashtable直接使用对象的`hashCode`。hashCode是JDK根据对象的地址或者字符串或者数字算出来的int类型的数值。然后再使用取余数获得最终的位置。
```java
int hash = key.hashCode();
int index = (hash & 0x7FFFFFFF) % tab.length;
```
Hashtable在计算元素的位置时需要进行一次除法运算，而除法运算是比较耗时的。

HashMap为了提高计算效率，将哈希表的大小固定为了2的幂，这样在取模预算时，不需要做除法，只需要做位运算。位运算比除法的效率要高很多。这样效率虽然提高了，但是hash冲突却也增加了。因为它得出的hash值的低位相同的概率比较高，为了解决这个问题，HashMap重新根据hashcode计算hash值后，又对hash值做了一些运算来打散数据。使得取得的位置更加分散，从而减少了hash冲突。当然了，为了高效，HashMap只做了一些简单的位处理。从而不至于把使用2 的幂次方带来的效率提升给抵消掉。
```java
static final int hash(Object key) {
    int h;
    return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
}
```

### 参考资料
![HashMap 的实现原理](http://wiki.jikexueyuan.com/project/java-collection/hashmap.html)
