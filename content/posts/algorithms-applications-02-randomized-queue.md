title: Randomized Queue with Reservoir Sampling
date: 2018-07-21
mathjax: true
categories:
- CS
tags:
- Software Engineer
- Java
- Algorithms
- Reservoir Sampling
- Fisher–Yates shuffle
---

This blog explains an apllication of randomized queue algorithms.
## Permutation client memory challenge
A client program `Permutation.java` that takes an integer k as a command-line argument; reads in a sequence of strings from standard input using `StdIn.readString()`; and prints exactly k of them, uniformly at random. Print each item from the sequence at most once.

More detail could be found at programming assignment [specification](http://coursera.cs.princeton.edu/algs4/assignments/queues.html) and [checklist](http://coursera.cs.princeton.edu/algs4/checklists/queues.html), codes available in [algs4/queues/src/](https://github.com/congchan/algs4/tree/master/queues/src).

<!-- more -->

### Randomized queue
For a randomized queue, the item removed is chosen **uniformly** at random from items in the data structure.

Each iterator must return the items in **uniformly** random order. The order of two or more iterators to the same randomized queue must be mutually independent; each iterator must maintain its own random order.

API:
```java
public class RandomizedQueue<Item> implements Iterable<Item> {
   public RandomizedQueue() {}                 // construct an empty randomized queue
   public boolean isEmpty() {}                 // is the randomized queue empty?
   public int size() {}                        // return the number of items on the randomized queue
   public void enqueue(Item item) {}           // add the item
   public Item dequeue() {}                    // remove and return a random item
   public Item sample() {}                     // return a random item (but do not remove it)
   public Iterator<Item> iterator() {}         // return an independent iterator over items in random order
   public static void main(String[] args) {}   // unit testing (optional)
}
```

### Solution
The bonu point is to use only one RandomizedQueue object of maximum size at most k.

More specifically, as the program sees a sequence of input, one at a time, the programe could only keep k items in memory, and they should be selected at random from the sequence. If we know the total number of items (n), then the solution is easy: select ten distinct indices i between 1 and n with equal probability, and keep the i-th elements. The challenge is that we do not know the input sequence length in advance.

The idea is when reading in input strings one by one, we maintain the RandomizedQueue with size at most k on the fly. If the RandomizedQueue is full, then we need to decide whether a new input string should be accepted or not. If it should be accepted, then one of the old elements must be kicked out of the queues. The key point here is how to make the decision.

The algorithms explain the mechanism: For a loop over n, swap item `[n]` with a random item in the range `[0]` through `[n]`. We store only the first k elements (`[0 : k-1]`) as that are all we need. Afterwards, when we get a new string (index `[n]`), we'll swap it with one of the first k strings for a given probability `P`, otherwise just discard it.

The [Reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) algorithms could solve the problem:
> Reservoir sampling is a family of randomized algorithms for randomly choosing a sample of k items from a list S containing n items, where n is either a very large or unknown number. Typically n is large enough that the list doesn't fit into main memory.

1. Keep the first k items in memory.
2. When the i-th item arrives (for $i>k$):
    * with probability $k/i$, keep the new item (discard an old one, selecting which to replace at random, each with chance $1/k$)
    * with probability $1-k/i$, keep the old items (ignore the new one)

code available: https://github.com/congchan/algs4/tree/master/queues/src

In our case of implementation, for a loop over n, swap item `[n]` with a random item in the range `[0]` through `[n]`. We store only the first k elements (`[0 : k-1]`) as that are all we need. Afterwards, when we get a new string (index `[n]`), we'll swap it with one of the first k strings for a given probability `P`, otherwise just discard it.

There is another shuffle method called [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) and its $O(n)$ version called Knuth shuffle which could shuffle a given sequence.

Test report:
```
Correctness:  43/43 tests passed
Memory:       106/105 tests passed
Timing:       136/136 tests passed

Aggregate score: 100.10%
[Compilation: 5%, API: 5%, Findbugs: 0%, PMD: 0%, Checkstyle: 0%, Correctness: 60%, Memory: 10%, Timing: 20%]

Test 3 (bonus): check that maximum size of any or Deque or RandomizedQueue object
                created is equal to k
  * filename = tale.txt, n = 138653, k = 5
  * filename = tale.txt, n = 138653, k = 50
  * filename = tale.txt, n = 138653, k = 500
  * filename = tale.txt, n = 138653, k = 5000
  * filename = tale.txt, n = 138653, k = 50000
==> passed

Total: 3/2 tests passed!
```
