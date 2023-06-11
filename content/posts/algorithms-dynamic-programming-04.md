---
title: Dynamic Programming 04 - 丑数
date: 2017-09-04
mathjax: true
author: "Cong Chan"
tags: ['Java', 'Algorithms', 'Dynamic Programming']
---
## 丑数
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。
<!-- more -->

要判断一个数是不是丑数, 不断地分别除以`2, 3, 5`，然后检查`num`是否到达`1`:
```java
public boolean isUgly(int num) {
    //(除数包括`4`可以让代码更简洁)
    for (int i=2; i<6 && num>0; i++)
        while (num % i == 0)
            num /= i;
    return num == 1;
}
```
如果要返回第`n`个丑数([leetcode原题](https://leetcode.com/problems/ugly-number-ii/)), 情况就稍微复杂点. 从动态规划的角度考虑, 对于一个较大的丑数`N`, 必定是由某个更小的丑数`M`乘以`2, 3, 5`其中一个得来的. 所以可以从小到大不断生成丑数. 为了避免在循环中每一次计算都从头开始检查每一个数`k`对应的`2*k, 3*k, 5*k`, 需要用三个变量`last2, last3, last5`来分别记录最近一次用到的丑数的索引, 下一次计算时就直接从上一次停止的地方开始运行.
```java
/** return the nth ugly number */
public static int unglyNumber(int n) {
    final int INIT = 5;
    int[] uglys = new int[n + INIT];
    for (int i = 0; i < 5;) {
        uglys[i] = ++i;
    }
    int last2, last3, last5, m2, m3, m5;
    last2 = last3 = last5 = 0;
    m2 = m3 = m5 = 1;
    for (int i = INIT; i < n; i++) {

        for (int j = last2 + 1; j < i; j++) {
            if (m2 <= uglys[i - 1] && uglys[j] * 2 > uglys[i - 1]) {
                m2 = uglys[j] * 2;
                last2 = j;
            }
        }

        for (int j = last3 + 1; j < i; j++) {
            if (m3 <= uglys[i - 1] && uglys[j] * 3 > uglys[i - 1]) {
                m3 = uglys[j] * 3;
                last3 = j;
            }
        }

        for (int j = last5 + 1; j < i; j++) {
            if (m5 <= uglys[i - 1] && uglys[j] * 5 > uglys[i - 1]) {
                m5 = uglys[j] * 5;
                last5 = j;
            }
        }

        uglys[i] = Math.min(Math.min(m2, m3), m5);
    }
    return uglys[n - 1];
}
```
[这里](https://www.geeksforgeeks.org/ugly-numbers/)提供了另一个理解这个问题的思路，并由此得出了一个更快的的算法(`O(n)`)：根据前面算法的原理，可以知道下一个丑数一定是前面某一个丑数乘以2,3,5中的一个，所以可以把问题转换为从以下三组数据中不断取最小值的问题：
```
(1) 1×2, 2×2, 3×2, 4×2, 5×2, …
(2) 1×3, 2×3, 3×3, 4×3, 5×3, …
(3) 1×5, 2×5, 3×5, 4×5, 5×5, …
```
可以发现每个子序列是丑数序列本身`1, 2, 3, 4, 5......`分别乘以`2, 3, 5`。使用类似merge sort的合并方法，每次从三个数组中弹出最小的数:
```java
/** return the nth ugly number */
public static int getUglyNumber(int n) {
    if (n == 0) return 0;
    int[] ugly = new int[n];
    ugly[0] = 1;
    int i2 = 0, i3 = 0, i5 = 0;
    int next2 = 2, next3 = 3, next5 = 5;
    for (int i = 1; i < n; i++)
    {
        ugly[i] = Math.min(next2, Math.min(next3, next5));
        if (next2 == ugly[i])
            next2 = ugly[++i2] * 2;
        if (next3 == ugly[i])
            next3 = ugly[++i3] * 3;
        if (next5 == ugly[i])
            next5 = ugly[++i5] * 5;
    }
    return ugly[n - 1];
}
```
