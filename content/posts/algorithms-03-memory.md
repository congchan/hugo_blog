---
title: Algorithms 03 - Memory 内存
date: 2017-06-28
mathjax: true
author: "Cong Chan"
tags: ['Software Engineer', 'Java', 'cs61b', 'algs4']
---
## Memory
Bit. 0 or 1.
Byte. 8 bits.
Megabyte (MB). 1 million or $2^{20}$ bytes.
Gigabyte (GB). 1 billion or $2^{30}$ bytes.
64-bit machine. We assume a 64-bit machine with 8 byte pointers (References).
・Can address more memory.
・Pointers use more space (some JVMs "compress" ordinary object pointers to 4 bytes to avoid this cost).
<!-- more -->

### Typical memory usage for primitive types and arrays
primitive types (bytes):
`boolean` 1
`byte` 1
`char` 2
`int` 4
`float` 4
`long` 8
`double` 8

for one-dimensional arrays (bytes):
`char[]` 2N + 24
`int[]` 4N + 24
`double[]` 8N + 24

### Typical memory usage for objects in Java
Object overhead. 16 bytes.
Reference. 8 bytes.
Padding. Each object uses a multiple of 8 bytes.
![](/images/string_memory.png "image from: https://www.coursera.org/learn/algorithms-part1/")
