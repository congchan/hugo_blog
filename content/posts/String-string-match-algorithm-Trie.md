title: 单模式匹配与拼写检查 - Trie
date: 2017-09-28
mathjax: true
categories:
- CS
tags:
- String
- Algorithms
- Trie
---
Trie 也称字典树，名称来源于Re<font color="red">trie</font>val，支持$O(n)$插入和查询操作，以空间换取时间的数据结构. 用于词频统计和输入统计领域, 可以高效地存储大规模的字典数据, 也可以用于模糊匹配, 搜索最长前缀词等.
> A **trie**, also called **digital tre**e, **radix tree** or **prefix tree** is a kind of search tree - an ordered tree data structure used to store a dynamic set or associative array where the keys are usually strings. Unlike a binary search tree, no node in the tree stores the key associated with that node; instead, its position in the tree defines the key with which it is associated. All the descendants of a node have a common prefix of the string associated with that node, and the root is associated with the empty string. Keys tend to be associated with leaves, though some inner nodes may correspond to keys of interest. Hence, keys are not necessarily associated with every node.

<!-- more -->
![](/images/Trie_example.png "A trie for keys "A","to", "tea", "ted", "ten", "i", "in", and "inn". Image from https://en.wikipedia.org/wiki/Trie")

## Trie
Trie没有规定每一个节点的分支数量, 用**R-way Trie**来表示分支数量为`R`的Trie. 对于不同的应用, 可以设置不同的`R`.

### 字符（模糊）匹配与拼写检查
应用例子是在一本字典中查找特定前缀的所有单词. 简化的例子是在英文字典中, 根据查询前缀, 返回相同前缀的所有单词数. 同样的结构可以用来检查拼写错误. 那么只需要在每一个节点存储该节点以下所有单词数就行了. 每一个节点包含一个长度26的数组，以方便快速定位对应的26个字母, 类似B-tree:![](/images/b_tree.png "image from https://www.coursera.org/learn/algorithms-part1")
```java
/**
 * from https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/TrieSET.java.html
 * @author Robert Sedgewick
 * @author Kevin Wayne
 */
public class TrieSET implements Iterable<String> {
    private static final int R = 256;        // extended ASCII

    private Node root;      // root of trie
    private int n;          // number of keys in trie

    // R-way trie node
    private static class Node {
        private Node[] next = new Node[R];
        private boolean isString;
    }

    /**
     * Initializes an empty set of strings.
     */
    public TrieSET() {
    }

    /**
     * Does the set contain the given key?
     * @param key the key
     * @return {@code true} if the set contains {@code key} and
     *     {@code false} otherwise
     * @throws IllegalArgumentException if {@code key} is {@code null}
     */
    public boolean contains(String key) {
        if (key == null) throw new IllegalArgumentException("argument to contains() is null");
        Node x = get(root, key, 0);
        if (x == null) return false;
        return x.isString;
    }

    private Node get(Node x, String key, int d) {
        if (x == null) return null;
        if (d == key.length()) return x;
        char c = key.charAt(d);
        return get(x.next[c], key, d+1);
    }

    /**
     * Adds the key to the set if it is not already present.
     * @param key the key to add
     * @throws IllegalArgumentException if {@code key} is {@code null}
     */
    public void add(String key) {
        if (key == null) throw new IllegalArgumentException("argument to add() is null");
        root = add(root, key, 0);
    }

    private Node add(Node x, String key, int d) {
        if (x == null) x = new Node();
        if (d == key.length()) {
            if (!x.isString) n++;
            x.isString = true;
        }
        else {
            char c = key.charAt(d);
            x.next[c] = add(x.next[c], key, d+1);
        }
        return x;
    }

    /**
     * Returns the number of strings in the set.
     * @return the number of strings in the set
     */
    public int size() {
        return n;
    }

    /**
     * Is the set empty?
     * @return {@code true} if the set is empty, and {@code false} otherwise
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Returns all of the keys in the set, as an iterator.
     * To iterate over all of the keys in a set named {@code set}, use the
     * foreach notation: {@code for (Key key : set)}.
     * @return an iterator to all of the keys in the set
     */
    public Iterator<String> iterator() {
        return keysWithPrefix("").iterator();
    }

    /**
     * Returns all of the keys in the set that start with {@code prefix}.
     * @param prefix the prefix
     * @return all of the keys in the set that start with {@code prefix},
     *     as an iterable
     */
    public Iterable<String> keysWithPrefix(String prefix) {
        Queue<String> results = new Queue<String>();
        Node x = get(root, prefix, 0);
        collect(x, new StringBuilder(prefix), results);
        return results;
    }

    private void collect(Node x, StringBuilder prefix, Queue<String> results) {
        if (x == null) return;
        if (x.isString) results.enqueue(prefix.toString());
        for (char c = 0; c < R; c++) {
            prefix.append(c);
            collect(x.next[c], prefix, results);
            prefix.deleteCharAt(prefix.length() - 1);
        }
    }

    /**
     * Returns all of the keys in the set that match {@code pattern},
     * where . symbol is treated as a wildcard character.
     * @param pattern the pattern
     * @return all of the keys in the set that match {@code pattern},
     *     as an iterable, where . is treated as a wildcard character.
     */  
    public Iterable<String> keysThatMatch(String pattern) {
        Queue<String> results = new Queue<String>();
        StringBuilder prefix = new StringBuilder();
        collect(root, prefix, pattern, results);
        return results;
    }
        
    private void collect(Node x, StringBuilder prefix, String pattern, Queue<String> results) {
        if (x == null) return;
        int d = prefix.length();
        if (d == pattern.length() && x.isString)
            results.enqueue(prefix.toString());
        if (d == pattern.length())
            return;
        char c = pattern.charAt(d);
        if (c == '.') {
            for (char ch = 0; ch < R; ch++) {
                prefix.append(ch);
                collect(x.next[ch], prefix, pattern, results);
                prefix.deleteCharAt(prefix.length() - 1);
            }
        }
        else {
            prefix.append(c);
            collect(x.next[c], prefix, pattern, results);
            prefix.deleteCharAt(prefix.length() - 1);
        }
    }

    /**
     * Returns the string in the set that is the longest prefix of {@code query},
     * or {@code null}, if no such string.
     * @param query the query string
     * @return the string in the set that is the longest prefix of {@code query},
     *     or {@code null} if no such string
     * @throws IllegalArgumentException if {@code query} is {@code null}
     */
    public String longestPrefixOf(String query) {
        if (query == null) throw new IllegalArgumentException("argument to longestPrefixOf() is null");
        int length = longestPrefixOf(root, query, 0, -1);
        if (length == -1) return null;
        return query.substring(0, length);
    }

    // returns the length of the longest string key in the subtrie
    // rooted at x that is a prefix of the query string,
    // assuming the first d character match and we have already
    // found a prefix match of length length
    private int longestPrefixOf(Node x, String query, int d, int length) {
        if (x == null) return length;
        if (x.isString) length = d;
        if (d == query.length()) return length;
        char c = query.charAt(d);
        return longestPrefixOf(x.next[c], query, d+1, length);
    }

    /**
     * Removes the key from the set if the key is present.
     * @param key the key
     * @throws IllegalArgumentException if {@code key} is {@code null}
     */
    public void delete(String key) {
        if (key == null) throw new IllegalArgumentException("argument to delete() is null");
        root = delete(root, key, 0);
    }

    private Node delete(Node x, String key, int d) {
        if (x == null) return null;
        if (d == key.length()) {
            if (x.isString) n--;
            x.isString = false;
        }
        else {
            char c = key.charAt(d);
            x.next[c] = delete(x.next[c], key, d+1);
        }

        // remove subtrie rooted at x if it is completely empty
        if (x.isString) return x;
        for (int c = 0; c < R; c++)
            if (x.next[c] != null)
                return x;
        return null;
    }

}
```

如果要问题扩展为返回所有相同前缀的单词，那么就要在插入字典时，在对应单词结尾的节点标记颜色。

### 提高扩展性
用固定长度为26的数组来处理英文，好处是数组内存占用小，索引时也不需要搜索，直接用字符码作为索引。也可以根据ASCII码进一步扩大数组长度以支持更多字符。

为了提高可扩展性，可以考虑用其他更灵活的数据结构来替代数组，比如HashMap，同时把HashMap放进一个TrieNode类。这样以后要修改核心的存储结构，只需要改动TrieNode即可，其余的接口不用改。
```java
public static class Trie {
    private TrieNode node;

    public Trie() {
        this.node = new TrieNode();
    }

    public void insert(String word) {
        TrieNode curNode = node;
        for (char x : word.toCharArray()) {
            curNode = curNode.set(x);
        }
    }

    public int search(String prefix) {
        TrieNode curNode = node;
        for (char x : prefix.toCharArray()) {
            if (curNode.get(x) == null) {
                return 0;
            }
            curNode = curNode.get(x);
        }
        return curNode.count;
    }

    public static class TrieNode {
        HashMap<Character, TrieNode> map;
        private int count;
        private char value;

        public TrieNode() {
            count = 0;
            map = new HashMap<>();
        }

        public TrieNode(Character val) {
            count = 1;
            this.value = val;
            map = new HashMap<>();
        }

        public TrieNode get(char ch) {
            return map.get(ch);
        }

        public TrieNode set(char ch) {
            TrieNode t = map.get(ch);
            if (t == null) {
                t = new TrieNode(ch);
                this.map.put(ch, t);
            } else {
                t.count++;
            }
            return t;
        }

        public int getCount() {
            return this.count;
        }

        public char getValue() {
            return this.value;
        }

    }

}
```
HashMap的寻址虽然会靠字符码作为地址的数组慢一点点，但也是非常快的:$O(\log N)$。但HashMap本身是比较耗内存的数据结构, 所以如果知道要处理的数据是在特定范围内的, 比如节点就是在256个字符中, 那么还是不要不用HashMap.
