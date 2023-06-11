---
title: 众数问题 - Boyer–Moore majority vote algorithm
date: 2017-10-03
mathjax: true
author: "Cong Chan"
tags: ['Java', 'Algorithms', 'Dynamic Programming']
---
数组中有一个数字出现的次数超过数组长度的一半，例如输入一个长度为9的数组`1,2,3,2,2,2,5,4,2`。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。因为这个数出现次数超过了数组长度一半以上, 那么它就是数组中出现次数最多的数, 故谓之**众数**.
<!-- more -->
```python
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        most = numbers[0]
        count = 1
        for item in numbers:
            if item == most:
                count += 1
            else:
                count -= 1
                if count < 0:
                    most = item
                    count = 1
        return 0 if numbers.count(most) <= len(numbers) / 2 else most
```

## 众数问题
众数问题可以推广泛化：给定大小为`n`的整数数组，找到所有出现超过`n / m`次的元素。这种问题可以使用 Boyer-Moore 算法解决.
> The Boyer–Moore majority vote algorithm is an algorithm for finding the majority of a sequence of elements using linear time and constant space. It is named after Robert S. Boyer and J Strother Moore, who published it in 1981, and is a prototypical example of a streaming algorithm.

如果存在众数元素，该算法会找到众数元素：对于出现次数一半以上的元素。但是，如果没有众数，算法将不会检测到该事实，并且仍将输出其中一个元素。

这个时候需要第二次遍历数据, 验证在第一次通过中找到的元素是否真正占众数。

比如找到所有出现超过`n / 3`次的元素, 最多只可能有2个, 可以用长度为2的数据结构(这里选择map)来记录众数.
```python
class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        m = 2
        cand = [0] * m
        freq = {}
        for item in nums:
            if len(freq) < m:
                freq[item] = 1 + freq.get(item, 0)
            elif item in freq:
                freq[item] += 1
            else:
                for k in list(freq):
                    freq[k] -= 1
                    if freq[k] <= 0:
                        freq.pop(k)

        return [k for k in freq if nums.count(k) > len(nums) // (m + 1)]
```
