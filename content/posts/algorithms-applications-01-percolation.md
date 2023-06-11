---
title: Percolations problem
date: 2018-07-03
mathjax: true
author: "Cong Chan"
tags: ['Software Engineer', 'Java', 'Algorithms', 'Bitwise Operation']
---
### Union-find applications: Percolation
[Problem discriptions](http://coursera.cs.princeton.edu/algs4/assignments/percolation.html)

Percolation data type. To model a percolation system, create a data type Percolation with the following API:
```java
public class Percolation {
    public Percolation(int n);                // create n-by-n grid, with all sites blocked
    public    void open(int row, int col);    // open site (row, col) if it is not open already
    public boolean isOpen(int row, int col);  // is site (row, col) open?
    public boolean isFull(int row, int col);  // is site (row, col) full?
    public     int numberOfOpenSites();       // number of open sites
    public boolean percolates();              // does the system percolate?
}
```
Monte Carlo simulation. To estimate the percolation threshold, consider the following computational experiment:
- Initialize all sites to be blocked.
- Repeat the following until the system percolates:
    - Choose a site uniformly at random among all blocked sites.
    - Open the site.
- The fraction of sites that are opened when the system percolates provides an estimate of the percolation threshold.

Codes available at [algs4/Percolation/src/](https://github.com/congchan/algs4)
#### [The back wash issue](http://coursera.cs.princeton.edu/algs4/checklists/percolation.html)
![](http://coursera.cs.princeton.edu/algs4/checklists/percolation-backwash.png "image from: http://coursera.cs.princeton.edu")
<!-- more -->
My solution inspired from this post https://www.sigmainfy.com/blog/avoid-backwash-in-percolation.html, with some improvements:
1. Using one `WeightedQuickUnionUF(n * n)` objects to track each site's parent.
2. Use a `byte[n * n]` to store the each site's state.
    - There are four possible states, represented as
        - `BLOCKED: 0b000`
        - `OPEN: 0b001`
        - `CONNECT_TO_BOTTOM: 0b010`
        - `CONNECT_TO_TOP: 0b100`
    - With byte operation `|`, we enable sites to have mixture of states.
3. `open(row, col)`: to open the current site `cur`, we need to
    - find out its four possible neibourghs (`up`, `down`, `left`, `right`, if exist);
    - use `find()` to return the neibourghs' parents (`upParent`, etc..), use `union()` to connect `cur` and its neibourghs;
    - Fianally, update `cur`'s new parent `newParent`'s state with the combination of `cur`'s parent state and the neibourghs' parents states.
    - in totalm, there involves 4 `union()` and 5 `find()` API calls at most but the time complexity is still $\Theta(\lg N)$
4. `isOpen()`: $\in \Theta(1)$ by checking the `byte[n * n]`.
5. `isFull()`: $\in \Theta(1)$, use one call `find()` API and thus is $\in \Theta (\lg N)$
6. `percolates()`: use a `boolean isPercolates` as mark, for any new open site that becomes both `CONNECT_TO_BOTTOM` and `CONNECT_TO_TOP`, we could mark the model as percolates.

```
Estimated student memory = 9.00 n^2 + 0.00 n + 160.00   (R^2 = 1.000)

Test 2 (bonus): check that total memory <= 11 n^2 + 128 n + 1024 bytes

==> passed
```
