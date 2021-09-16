# Index Tracking Description

More information about `indextrack` datasets can be found [here](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/indtrackinfo.html).

There are currently 8 data files.

Five of these data files are the test problems used in the paper:
"An evolutionary heuristic for the index tracking problem", 
J.E. Beasley, N. Meade and T.-J. Chang, 
European Journal of Operational Research, vol. 148, 2003, pp621-643

The test problems are the files:
`indtrack1`, `indtrack2`, ..., `indtrack5`

The format of these data files is:
```
number of stocks (N), number of time periods (T)
index value at time t (t=0,1,...,T)
for each stock i (i=1,...,N) in turn:
     stock value at time t (t=0,1,...,T)
```

Note here that time runs from 0 to T (i.e. there are T+1
time periods in total)

The largest file is indtrack5 of size 930Kb (approximately)
The entire set of files is of size 2.1Mb (approximately).

Three further data files, for larger test problems were added in April 2007
These have the same format as the files above and are:
`indtrack6`, `indtrack7` and `indtrack8`.

These three files have been used in:

Mixed-integer programming approaches for index tracking and 
enhanced indexation (N.A.Canakgoz and J.E. Beasley) 
European Journal of Operational Research vol. 196, 2009, pp384-399

### Names of Datasets

(TODO: Find the dataset date ranges)

1. Hang Seng
2. DAX 100
3. FTSE 100
4. S&P 100
5. Nikkei 225
6. S&P 500
7. Russell 2000
8. Russell 3000