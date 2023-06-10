title: 字符串符号表和三元搜索Trie
date: 2017-09-30
mathjax: true
categories:
- CS
tags:
- String
- Algorithms
- Information Retrieval
- Symbol table
- Trie
---

## 符号表
> 在计算机科学中，符号表是一种用于语言翻译器（例如编译器和解释器）中的数据结构。在符号表中，程序源代码中的每个标识符都和它的声明或使用信息绑定在一起，比如其数据类型、作用域以及内存地址。
常用哈希表来实现.

符号表的应用非常广泛, 可用于实现Set, Dictionary, 文件索引, 稀疏向量/矩阵等数据结构和相关的运算操作, 还有其他如过滤查询(Exception filter), 一致性查询(concordance queries)等操作.

字符符号表就是专门针对字符操作的符号表, API:
Prefix match - Keys with prefix `sh`: `she`, `shells`, and `shore`.
Wildcard match - Keys that match `.he`: `she` and `the`.
Longest prefix - Key that is the longest prefix of `shellsort`: `shells`.
<!-- more -->
```java
public interface StringST<Value> {
    StringST(); create a symbol table with string keys
    void put(String key, Value val); put key-value pair into the symbol table
    Value get(String key); value paired with key
    void delete(String key); delete key and corresponding value
    Iterable<String> keys(); all keys
    Iterable<String> keysWithPrefix(String s); keys having s as a prefix
    Iterable<String> keysThatMatch(String s); keys that match s (where . is a wildcard)
    String longestPrefixOf(String s); longest key that is a prefix of s
}
```

### 以Trie为基础的字符符号表
algs4中提供了用 R-way trie 来实现符号表(symbol table)例子:
```java
public class TrieST<Value> {
    private static final int R = 256; // extended ASCII
    private Node root = new Node();

    private static class Node {
        private Object value;
        private Node[] next = new Node[R];
    }

    public void put(String key, Value val) {
        root = put(root, key, val, 0);
    }

    private Node put(Node x, String key, Value val, int d) {
        if (x == null) x = new Node();
        if (d == key.length()) { x.value = val; return x; }

        char c = key.charAt(d);
        x.next[c] = put(x.next[c], key, val, d+1);
        return x;
    }

    public boolean contains(String key) { return get(key) != null; }

    public Value get(String key) {
        Node x = get(root, key, 0);
        if (x == null) return null;
        return (Value) x.val;
    }

    private Node get(Node x, String key, int d) {
        if (x == null) return null;
        if (d == key.length()) return x;
        char c = key.charAt(d);
        return get(x.next[c], key, d+1);
    }

}
```
按顺序迭代所有键：
·中序遍历trie，找到的键添加到队列中
·维护从根到当前节点路径的字符序列
```java
public Iterable<String> keys()
{
    Queue<String> queue = new Queue<String>();
    collect(root, "", queue);
    return queue;
}

private void collect(Node x, String prefix, Queue<String> q)
{
    if (x == null) return;
    if (x.val != null) q.enqueue(prefix);
    for (char c = 0; c < R; c++)
        collect(x.next[c], prefix + c, q);
}
```

### 前缀匹配
Find all keys in a symbol table starting with a given prefix.
Ex. Autocomplete in a cell phone, search bar, text editor, or shell.
・User types characters one at a time.
・System reports all matching strings.
```java
public Iterable<String> keysWithPrefix(String prefix)
{
    Queue<String> queue = new Queue<String>();
    Node x = get(root, prefix, 0);
    collect(x, prefix, queue);
    return queue;
}
```

### 最长前缀
Find longest key in symbol table that is a prefix of query string.
Ex. To send packet toward destination IP address, router chooses IP address in routing table that is longest prefix match.

・Search for query string.
・Keep track of longest key encountered.
```java
public String longestPrefixOf(String query)
{
    int length = search(root, query, 0, 0);
    return query.substring(0, length);
}

private int search(Node x, String query, int d, int length)
{
    if (x == null) return length;
    if (x.val != null) length = d;
    if (d == query.length()) return length;
    char c = query.charAt(d);
    return search(x.next[c], query, d+1, length);
}
```
