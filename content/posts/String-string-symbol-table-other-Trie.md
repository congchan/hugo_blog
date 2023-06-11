---
title: 不同树结构的字符串符号表
date: 2017-10-01
mathjax: true
author: "Cong Chan"
tags: ['String', 'Algorithms', 'Information Retrieval', 'Symbol table', 'Trie']
---
## 各种树的变种
为了适应不同的应用场景, 人们使用不同的树结构来实现符号表.

### 九宫格输入法
对于手机的九宫格输入法, 简单的实现方式是多次敲击: 通过反复按键输入一个字母，直到出现所需的字母。
<!-- more -->
但 http://www.t9.com/ 的 T9 texting 支持更高效的输入方法:
・Find all words that correspond to given sequence of numbers.
・Press 0 to see all completion options.
![](/images/t9.png)
Ex. hello
・多次敲击: 4 4 3 3 5 5 5 5 5 5 6 6 6
・T9: 4 3 5 5 6

可以使用 8-way trie 来实现.

### 三元搜索Trie
`R`较大的R-way trie的空间效率不高，读取比较大的文件往往导致内存不足。但弊端是开辟出的数组内存利用率其实不高。现在很多系统都使用Unicode，分支可高达`65,536`. 所以需要更高效的方法。

Ternary search tries:
・Store characters and values in nodes (not keys).
・Each node has 3 children: smaller (left), equal (middle), larger (right).
![](/images/tst.png "image from: https://www.coursera.org/learn/algorithms-part2/")
Search in a TST: Follow links corresponding to each character in the key.
・If less, take left link; if greater, take right link.
・If equal, take the middle link and move to the next key character.
```java
public class TST<Value>
{
    private Node root;
    private class Node
    {
        private Value val;
        private char c;
        private Node left, mid, right;
    }

    public void put(String key, Value val)
    { root = put(root, key, val, 0); }

    private Node put(Node x, String key, Value val, int d)
    {
        char c = key.charAt(d);
        if (x == null) { x = new Node(); x.c = c; }
        if (c < x.c) x.left = put(x.left, key, val, d);
        else if (c > x.c) x.right = put(x.right, key, val, d);
        else if (d < key.length() - 1) x.mid = put(x.mid, key, val, d+1);
        else x.val = val;
        return x;
    }

    public boolean contains(String key)
    { return get(key) != null; }

    public Value get(String key)
    {
        Node x = get(root, key, 0);
        if (x == null) return null;
        return x.val;
    }

    private Node get(Node x, String key, int d)
    {
        if (x == null) return null;
         char c = key.charAt(d);
         if (c < x.c) return get(x.left, key, d);
         else if (c > x.c) return get(x.right, key, d);
         else if (d < key.length() - 1) return get(x.mid, key, d+1);
         else return x;
    }
}
```
TSTs比hashing更快（特别是对于搜索缺失键的情况）。

### 基数树
Radix Tree, 也叫 Patricia trie (Practical Algorithm to Retrieve Information Coded in Alphanumeric), crit-bit tree, 压缩前缀树:
・Remove one-way branching.
・Each node represents a sequence of characters.
・Implementation: one step beyond this course.
![](/images/radix_trie.png "put("shells", 1); put("shellfish", 2); image from: https://algs4.cs.princeton.edu/")
对于基数树的每个节点，如果该节点是唯一的子树的话，就和父节点合并。

Applications.
・Database search.
・P2P network search.
・IP routing tables: find longest prefix match.
・Compressed quad-tree for N-body simulation.
・Efficiently storing and querying XML documents.

### 后缀树
后缀树（Suffix tree）指字符串后缀的基数树: 一个`String S`的后缀树是一个边（edge）被标记为字符串的树。因此每一个`S`的后缀都唯一对应一条从根节点到叶节点的路径。这样就形成了一个`S`的后缀的基数树。![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Suffix_tree_BANANA.svg/250px-Suffix_tree_BANANA.svg.png "image from: https://en.wikipedia.org/")

Applications.
・Linear-time: longest repeated substring, longest common substring, longest palindromic substring, substring search, tandem repeats, ….
・Computational biology databases (BLAST, FASTA).

## 字符符号表总结
Red-black BST.
・Performance guarantee: log N key compares.
・Supports ordered symbol table API.

Hash tables.
・Performance guarantee: constant number of probes.
・Requires good hash function for key type.

Tries. R-way, TST.
・Performance guarantee: log N characters accessed.
・Supports character-based operations.
![](/images/string_symbol_table_cost_sum.png "image from: https://www.coursera.org/learn/algorithms-part2/")
> You can get at anything by examining 50-100 bits
