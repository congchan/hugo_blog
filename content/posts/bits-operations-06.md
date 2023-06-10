title: 位操作 - 汉明距离
date: 2017-09-24
mathjax: true
categories:
- CS
tags:
- Bitwise Operation
- Software Engineer
- Computer Science
- Algorithms
---
求两个整数的汉明距离 hamming distance
<!-- more -->
[Leetcode 461](https://leetcode.com/problems/hamming-distance/description/)
两个整数之间的汉明距离是该两个数之间不同的位数。 给定两个整数x和y，计算汉明距离。问题也可以理解为对于两个整数`m`和`n`, 需要改变`m`的二进制多少位才能得到`n`:
```java
/** Use Brian Kernighan's way to count bits */
public int hammingDistance(int x, int y) {
    x = x ^ y;
    y = 0;
    while(x != 0){
        y++;
        x &= x - 1;
    }
    return y;
}
```
```java
public class Solution {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }
}
```
同样用到Brian Kernighan算法：
> Lets say that the bit at index n is 1 and that the bits in indexes 0 up to `n-1` are all 0 (we'll use little endianess - so index 0 is 1, index 1 is 2, index 2 is 4, index 3 is 8 and so on).

> `v-1` subtracts from index 0 - but it's 0, so it converts it to 1 and subtracts from index 1 - but it's also 0, so it converts it to 1 and subtracts from index 2 - and so on until we reach index n. Since index n is 1 it can subtract from it and turn it to 0 - and there it stops: `1101000 - 1 = 1100111`

> So, `v-1` is like `v` except there are `n` 0 that became 1 and one 1 that became 0. In `v & v - 1` all the other bits remain as is, the `n` zeros that where turned to ones remain 0 (because `0 & 1 == 0`), and the one 1 that was turned to 0 turns to 0(because `1 & 0 == 0`). So overall - only a single bit was changed in the iteration, and this change was from 1 to 0: `1101000 & 1100111 = 1100000`
