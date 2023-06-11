---
title: Dynamic Programming 03 - 最长公共子序列
date: 2017-09-03
mathjax: true
author: "Cong Chan"
tags: ['Java', 'Algorithms', 'Dynamic Programming']
---
## 最长公共子序列
对于一个字符串, 它的子序列，就是将给字符串中任意个元素去掉之后剩余的字符串, 所以子序列不要求是连续的, 但是维持原来的顺序. 在文本相似度比较中，常用到最长公共子序列（longest common sequence）。
<!-- more -->

同时遍历两个字符串, 如果`x[i] == y[j]`, 则`x[i]`和`y[j]`参与了最长公共子序列`z[k]`的构建.

如果用`lcs[i, j]`表示遍历到`x[0-i]`和`y[0-j]`时的LCS长度, 那么现在就需要判断`x[i]`和`y[j]`的关系, 分两种情况:
* 如果二者相等, 那么`lcs1 = lcs[i - 1, j - 1] + 1`
* 若不相等, 那么只能在`x`和`y`中选择一个进行推进, 选择依据就是取较大值, `lcs2 = max(lcs[i - 1, j], lcs[i, j - 1])`

初始状态自然是`lcs[0, 0] = 0`.
```java
static int[][] lcs;
public static int longestCS(String x, String y) {
    char[] xList = x.toCharArray();
    char[] yList = y.toCharArray();
    for (int i = 1; i <= xList.length; i++) {
        for (int j = 1; j <= yList.length; j++) {
            if (xList[i - 1] == yList[j - 1]) {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
            } else {
                lcs[i][j] = Math.max(lcs[i - 1][j], lcs[i][j - 1]);
            }
        }
    }

    return lcs[x.length()][y.length()];
}
```

### 最长公共子串
最长公共子串（longest common substring）, 要求的是任意连续的子字符串。设定`LCS(i, j)`为包含当前字符`a[i]`和`b[j]`的最长lcs. 假如当前满足`a[i] == b[j]`, 那么`LCS(i, j) = LCS(i - 1, j - 1) + 1`, 否则为0.

比如字符串`21232523311324`和字符串`312123223445`的匹配矩阵，前者为X方向的，后者为Y方向的。例子来源于[这篇文章](http://www.cnblogs.com/dartagnan/archive/2011/10/06/2199764.html)
```
0 0 0 1 0 0 0 1 1 0 0 1 0 0 0
0 1 0 0 0 0 0 0 0 2 1 0 0 0 0
1 0 2 0 1 0 1 0 0 0 0 0 1 0 0
0 2 0 0 0 0 0 0 0 1 1 0 0 0 0
1 0 3 0 1 0 1 0 0 0 0 0 1 0 0
0 0 0 4 0 0 0 2 1 0 0 1 0 0 0
1 0 1 0 5 0 1 0 0 0 0 0 2 0 0
1 0 1 0 1 0 1 0 0 0 0 0 1 0 0
0 0 0 2 0 0 0 2 1 0 0 1 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```
```java
public int findLongest(String A, int n, String B, int m) {
    char[] cha = A.toCharArray();
    char[] chb = B.toCharArray();
    int[][] matrix = new int[n][m];
    int max = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (cha[i] == chb[j])
            {
                int cur = (i > 0 && j > 0) ? matrix[i - 1][j - 1] + 1 : 1;
                max = Math.max(max, cur);
                matrix[i][j] = cur;
            }
    return max;
}
```
从另一个角度理解, 这个解法就是用一个矩阵来记录两个字符串中所有位置的两个字符之间的匹配情况，若是匹配则赋上其左上角元素的值加1，否则为0。矩阵中值最大的值，就对应着对角线最长的非0连续序列，其对应的位置就是最长匹配子串的位置，最长匹配子串的位置和长度就已经出来了。计算这个矩阵的复杂度是`O(N*M)`.
