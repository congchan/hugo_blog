title: 位操作 - 基础的位运算
date: 2017-08-30
mathjax: true
categories:
- CS
tags:
- Bitwise Operation
- Software Engineer
- Computer Science
- Algorithms
---
一些常规的操作， 参考[这个视频](https://www.youtube.com/watch?v=7jkIUgLC29I)。
<!-- more -->

## 基本位操作
把某一位变为`1`：
```python
def set_bit(x, position):
    mask = 1 << position
    return x | mask

bin(set_bit(0b110, 0b101))
```
输出`0b100110`. 因为`x = 0b110 = 6`, 翻转第五位，就用`position = 0b101 = 5`， 得到`mask = 0b00100000`, 用`|`把第五位变为`1`.

清除某一位（`1`变`0`)：
```python
def clear_bit(x, position):
    mask = 1 << position
    return x & ~mask
```

通过XOR`^`和`1`来翻转某一位：
```python
def flip_bit(x, position):
    mask = 1 << position
    return x ^ mask
```

通过`&1`可以作为取位操作, 来判断某一位是否是`1`:
```python
def is_bit_set(x, position):
    shifted = x >> position
    return shifted & 1
```
`0b1100110 >> 0b101` = `0b11`, `0b11 & 0b01` = `1`

根据参数`state`来控制修改某一位, 如果参数是`1`那么就是set, 如果是`0`那么就是clear:
```python
def modify_bit(x, position, state):
    mask = 1 << position
    return (x & ~mask) | (-state & mask)
```
如果`state = 0b1`, `-state = 0b11111111`
如果`state = 0b0`, `-state = 0b0`
