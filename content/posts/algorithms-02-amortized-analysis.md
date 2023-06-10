title: Algorithms 02 - Amortized Analysis 平摊分析
date: 2017-06-27
mathjax: true
categories:
- CS
tags:
- Software Engineer
- Java
- cs61b
- algs4
---
假如有两种交税方式：
1. 每天付 3 金币
2. 每次付的金币呈指数级增长，但通知付款频率呈指数级下降
    - 第1天：付 1
    - 第2天：付 2 (累计 3)
    - 第4天：付 4 (累积 7)
    - 第8天：付 8 (累积 15)

哪种付的钱比较少？
第二种比较划算，本质上等同于每天付 2，就是**amortized constant**。
<!-- more -->
A more rigorous examination of amortized analysis is done here, in three steps:
1. Pick a cost model (like in regular runtime analysis)
2. Compute the average cost of the i'th operation
3. Show that this average (amortized) cost is bounded by a constant.

类似的应用在[Array list 扩容](/NOTE-CS61B-data-structures-07-java-array-based-list#数组扩容)中提到的 geometric resizing 方法(实际也是Python list 使用的方法)有体现, 所以使用一个因数来扩容数组, 可以让 ArrayList 的 `add`操作变为 amortized constant time.

总结
>- Amortized analysis provides a way to prove the average cost of operations.
>- If we chose $a_i$ such that $\Phi_i$ is never negative and $a_i$ is constant for all $i$, then the amortized cost is an upper bound on the true cost.
-- from: https://joshhug.gitbooks.io/
