title: Dynamic Programming 01 - 理解动态规划
date: 2017-09-01
mathjax: true
categories:
- CS
tags:
- Java
- Algorithms
- Dynamic Programming
---
我们已经看到了一些优雅的设计原则，例如分而治之，图探索和贪婪的选择，它们为各种重要的计算任务提供了确定的算法。这些工具的缺点是只能用于非常特定类型的问题。现在我们来谈谈算法工艺的两个大锤，即动态编程和线性编程，这两种适用性非常广泛的技术可以在更专门的方法失败时调用。可以预见的是，这种普遍性通常会带来效率上的损失。

很多经典的方法，如 divide-and-conquer, graph exploration, and greedy等, 为各种重要的计算任务提供了确定性的解法。但是这些算法只能用于特定类型的问题。

这里介绍两个算法大杀器: **Dynamic programing** 和 **linear programming**.

这两种适用性非常广的算法可以在黔驴技穷时考虑调用（如 the knapsack problem, sequence alignment, and optimal binary search trees）。当然，普遍性往往会带来效率上的损失。

<!-- more -->

## 动态规划
动态规划作为一种编程范例，可以从一个例子着手理解：求数列的 maximum-weighted independent sets (MIS, 最大非连续非相邻子集)和, 对于`a = [1, 4, 5, 4]`, 其MIS为`{a[1], a[3]} = 8`.

如果使用贪心法, 每次都在可选范围内取最大值, 那么就会得到`{a[2], a[0]} = 6`.

如果使用分而治之法, 把数组分为两半`a1 = [1, 4], a2 = [5, 4]`, 则分别得到MIS`{a1[1]}, {a2[0]}`, 合并后发现是相邻的, 与要求相悖.

要解决这个问题，关键的步骤是找到**基于子问题最优解的最优解**：想办法把缩小最优解备选方案的数量，在这个较小的空间中可以直接采取暴力搜索寻找最优解。

对于`a = [1, 4, 5, 4]`, 假设其MIS为`S`, 假如从最右边的元素开始考虑, `a[3] = 4`只有属于`S`和不属于`S`两种情况
* 若`a[3] = 4`属于`S`: 那么`a[2] = 5`就肯定不属于`S`, 则`S1 = MIS([1, 4]) + MIS([4])`
* 若`a[3]`不属于`S`: 那么`S`只能存在于`[1, 4, 5]`中, 问题就变成`S2 = MIS([1, 4, 5])`

所以归纳出`S = max(S1, S2) = max(MIS([1, 4]) + MIS([4]), MIS([1, 4, 5])) = ...`。 对于只剩下一个元素的去情况, `MIS([4]) = max(4) = 4`, 即`MIS([i]) = i`

这就是一个递归的调用: 也就是从右往左, 每一个元素都要考虑一遍是否属于`S`, 每次会分裂出两种情况, 所以递归的复杂度是$Θ(2^n)$.

这个算法效率不高, 需要优化. 我们考虑这个问题到底有多少不同的子问题? 因为我们是从右往左扫描每一个元素, 对于每一个元素`i`, 不管其属于或不属于`S`, 待解决的递归子问题只有一个, 就是求其左边的所有元素(前缀)的MIS, 所以理论上有$Θ(n)$个不同的子问题.

所以虽然递归的调用是$Θ(2^n)$, 但需要解决的子问题只有$Θ(n)$, 那么就存在优化的空间. 办法就是通过**记忆**已经解决了的子问题的答案, 来避免重复的计算. 因为右边的元素的子问题答案需要用到其左边的子问题的答案, 所以计算时, 要从左往右计算. 

定义子问题: 用`MIS(i)`表示`a[i]`的前缀`a[: i]`的MIS(不包括`a[i]`),
1. `MIS(0) = 0`, 因为`a[0]`左边没有任何元素.
2. `MIS(1) = max(a[0:1]) = a[0]`
3. 对于`i > 1`, `MIS(i)`只有两种情况, 取二者中较大者:
   1. 包括`a[i - 1]`, `MIS(i) = MIS(i - 2) + a[i - 1]`
   2. 不包括, `MIS(i) = MIS(i - 1)`

从开头开始考虑(计算), 每一步`i`都记住对应子问题的最优解`MIS(i)`, 计算到最后一个子问题`MIS(N)`, 就得出了考虑了所有子问题之后的最大值

```java
public static int[] forwardMIS(int[] a) {
    int[] mis = new int[a.length + 1];
    mis[0] = 0;
    mis[1] = a[0];
    for (int i = 2; i < mis.length; i++) {
        mis[i] = Math.max(mis[i - 1], mis[i - 2] + a[i - 1]);
    }
    return mis;
}
```

但以上算法并没有记录MIS具体包含哪些子集，虽然可以通过修改`mis`数据结构来额外存储每个值对应的MIS点集, 但这样会影响效率而且浪费内存空间.

回忆前面从右往左的分析, 每个元素都会考量是否属于MIS`S`, 所以我们可以把`forwardMIS`中算好的`mis`数组从右往左依次判断一遍, 已决定是否把`a`对应位置的元素加入到`S`中.
```java
public static ArrayList<Integer> backwardMIS(int[] a) {
    ArrayList<Integer> s = new ArrayList<>();
    int i = mis.length - 1;
    while (i >= 2) {
        if (mis[i - 1] >= mis[i - 2] + a[i - 1]) {
            i--;
        } else {
            s.add(a[i - 1]);
            i -= 2;
        }
    }
    return s;
}
```
进一步优化，我们可以用类似`backward`的算法一次过(`O(n)`)计算出MIS的集合和MIS的值, 对`backward`算法稍作改动, 在`a`表末尾延申一个`0`元素, 然后从右往左依次判断一遍, 直接用`a`表自身的值来判断, 得到一个Backward算法:
```java
static int sum = 0;
public static ArrayList<Integer> backwardMISI(int[] a) {
    ArrayList<Integer> s = new ArrayList<>();
    int i = a.length - 2; // assume a = [a, 0]
    while (i >= 0) {
        int x = get(a, i);
        if (get(a, i + 1) >= get(a, i + 2) + x) {
            i--;
        } else {
            s.add(x);
            sum += x;
            i -= 2;
        }
    }
    return s;
}

private static int get(int[] a, int index) {
    if (index > a.length -1) {
        return 0;
    } else {
        return a[index];
    }
}
```
总结动态规划的解法：
* 定义合适的子问题集合: 这些子问题应该尽可能小，数量尽可能少。因为即使在最好的情况下，也要花费 constant time 来解决每个子问题，因此子问题的数量和大小就是整个算法运行时间的下限。
* 归纳转移方程：系统地解决从最小的子问题开始的所有子问题后，如何转向越来越大的子问题。
* 通过**记忆**减少重复的递归调用计算: 要求前面子问题的解决方案能够用来快速计算当前子问题。

### 参考资料
* https://people.eecs.berkeley.edu/~vazirani/algorithms/chap6.pdf
* http://cs.yazd.ac.ir/farshi/Teaching/DP3901/
