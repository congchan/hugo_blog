---
title: Dynamic Programming 02 - 最大子序列
date: 2017-09-02
mathjax: true
author: "Cong Chan"
tags: ['Java', 'Algorithms', 'Dynamic Programming']
---
## 最大子序列
> Maximum subarray problem: In computer science, the maximum subarray problem is the task of finding the contiguous subarray within a one-dimensional array of numbers which has the largest sum. For example, for the sequence of values −2, 1, −3, 4, −1, 2, 1, −5, 4; the contiguous subarray with the largest sum is 4, −1, 2, 1, with sum 6.
The problem was first posed by Ulf Grenander of Brown University in 1977, as a simplified model for maximum likelihood estimation of patterns in digitized images. A linear time algorithm was found soon afterwards by Jay Kadane of Carnegie Mellon University (Bentley 1984).

<!-- more -->
如果我们知道在位置`i`结束的最大`MSP[i]`，那么在位置`i + 1`处结束的`MSP[i+1]`就是有两种情况，一是包含`MSP[i]`，即`MSP[i+1] = MSP[i] + nums[i]`，二是不包含, 即`MSP[i+1] = nums[i]`, 二者取较大值。
```java
public int maxSubArray(int[] nums) {
    int maxCur = nums[0]; // maximum value contains current
    int maxSoFar = maxCur; // maximum value found so far
    for(int i = 1; i < nums.length; i++) {
        maxCur = Math.max(nums[i], nums[i] + maxCur);
        maxSoFar = Math.max(maxCur, maxSoFar);
    }
    return maxSoFar;
}
```

### 股票最优买卖时间点
给出一段股票价格变化序列，一次交易可以获得的最大收益。如`[7,1,5,3,6,4]`输出`5`, 在第2天买入（价格= 1）并在第5天卖出（价格= 6），利润 6-1 = 5。

解决逻辑跟最大子序列问题一样, 使用**Kadane's Algorithm**. 计算原始数组的差分, 并寻找给出最大利润的连续子序列, 如果差分小于0, 重置为0:
```java
public int maxProfit(int[] prices) {
    int maxCur = 0; // current maximum value
    int maxSoFar = 0; // maximum value found so far
    for(int i = 1; i < prices.length; i++) {
        maxCur = Math.max(0, maxCur + prices[i] - prices[i-1]);
        maxSoFar = Math.max(maxCur, maxSoFar);
    }
    return maxSoFar;
}
```

### 股票最优买卖时间点II
以上问题如果是无限次交易(但不允许在同一天内买卖股票):
```java
public int maxProfit(int[] prices) {
    int profit = 0, i = 0;
    while (i < prices.length) {
        // find next local minimum
        while (i < prices.length-1 && prices[i+1] <= prices[i])
            i++;
        int min = prices[i++]; // need increment to avoid infinite loop for "[1]"

        // find next local maximum
        while (i < prices.length-1 && prices[i+1] >= prices[i])
            i++;
        profit += i < prices.length ? prices[i++] - min : 0;
    }
    return profit;
}
```
如果允许T+0交易, 那么直接贪心求和所有正的差分项就好了.

### 股票最优买卖时间点III
参考[这个答案](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution)
以上问题如果是交易次数最多两次, 需要寻找动态规划转移方程. `profit[k, i]`表示第`k`次交易, 在`i`天的利润. 如果当天不交易, 那么利润不变`profit[k, i] = profit[k, i - 1]`. 如果当天卖出, 卖出的是第`j`天买入的股票(`j < i`), 那么利润就是`prices[i] - prices[j] + profit[k-1, j-1] `, 也就是要`prices[j] - profit[k-1, j-1]`最小.
```java
public int maxProfit(int[] prices) {
    if (prices.length == 0) return 0;
    int[][] profit = new int[3][prices.length];
    for (int k = 1; k <= 2; k++) {
        for (int i = 1; i < prices.length; i++) {
            int min = prices[0];
            for (int j = 1; j <= i; j++)
                min = Math.min(min, prices[j] - profit[k-1][j-1]);
            profit[k][i] = Math.max(profit[k][i-1], prices[i] - min);
        }
    }

    return profit[2][prices.length - 1];
}
```
因为`i`从左往右, `j < i`, 所有`min`不需要每次都从头开始找:
```java
public int maxProfit(int[] prices) {
    if (prices.length == 0) return 0;
    int[][] profit = new int[3][prices.length];
    for (int k = 1; k <= 2; k++) {
        int min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            min = Math.min(min, prices[i] - profit[k-1][i-1]);
            profit[k][i] = Math.max(profit[k][i-1], prices[i] - min);
        }
    }

    return profit[2][prices.length - 1];
}
```
复杂度为O(NK).
从循环上可以看到, `i`只依赖于`i-1`, `k`只依赖于`k-1`, 因此可以压缩为一维的数组来存储, 但需要改变交换`i`和`k`的循环
```java
public int maxProfit(int[] prices) {
    if (prices.length == 0) return 0;
    int[] profit = new int[3];
    int[] min = new int[3];
    for (int i = 1; i < prices.length; i++)  {
        for (int k = 1; k <= 2; k++) {
            min[k] = Math.Min(min[k], prices[i] - profit[k-1]);
            profit[k] = Math.Max(profit[k], prices[i] - min[k]);
        }
    }

    return profit[2];
}
```
因为在这里`k=2`, 所以可以使用有限个变量来储存状态:
```java
public int maxProfit(int[] prices)  {
    int buyOne = Integer.MAX_VALUE;
    int SellOne = 0;
    int buyTwo = Integer.MAX_VALUE;
    int SellTwo = 0;
    for(int p : prices) {
        buyOne = Math.min(buyOne, p);
        SellOne = Math.max(SellOne, p - buyOne);
        buyTwo = Math.min(buyTwo, p - SellOne);
        SellTwo = Math.max(SellTwo, p - buyTwo);
    }
    return SellTwo;
}
```
