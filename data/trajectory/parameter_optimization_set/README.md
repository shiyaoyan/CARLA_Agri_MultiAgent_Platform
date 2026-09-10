# Parameter optimization set

Place complete tractor trajectory CSV files used by Bayesian parameter
optimization in this directory. The platform scans only top-level `.csv`
files. Do not place independent-test trajectories here.

Each CSV should contain, or be convertible by PGA to, the standard fields:

```text
longitude,latitude,time,speed,direction,forward_reverse_label
```

