# C-Tarone: Finding Statistically Significant Interactions between Continuous Features
Source code of *C-Tarone*, which finds statistically significant multiplicative feature interactions from multivariate data.  
Please see the following paper for more details:  
* Sugiyama, M., Borgwardt, K.: **Finding Statistically Significant Interactions between Continuous Features**, IJCAI-19, 3490-3498 (2019).


## Usage
*C-Tarone* is written in C++11.
To compile it, the Boost library is required.

For example, in the directory `src/cc`:
```
$ make
$ ./ctarone -i synth_N=1000_n=20.dat -c synth_N=1000_05.class -o out -t stat
> Reading a database file     "synth_N=1000_n=20.dat" ... end
> Reading a class file        "synth_N=1000_05.class" ... end
  Sample size in total:       1000
  Sample size in class 0:     500
  # features:                 20
> Start enumeration of testable combinations ... end
  # testable combinations:    37910
  Corrected alpha:            1.31891e-06
  Frequency threshold:        0.0166722
  Running time:               0.456451 [sec]
> Find significant combinations with a threshold 0.0166722
  # significant combinations: 5302
  Running time:               0.490904 [sec]
```

### Arguments
* `-i <input_file>`: a path to a csv file of an input dataset (without row and column names)
* `-c <class_file>`: a path to a file of input class labels (each line represents a binary (0/1) label of the corresponding line in `<input_file>`)
* `-o <output_file>`: Output of significant feature combinations is written to `<output_file>`
* `-t <output_stat_file>`: Output of statistics is written to `<output_stat_file>`
* `-a <alpha>`: significance level for the FWER, where *C-Tarone* always guarantees "FWER < `<alpha>`"
* `-k <size_limit>`: the upper bound of the size of feature combinations (default: unlimited)
* `-v <vervose>`: Verbose mode if specified


## Contact
Author: Mahito Sugiyama  
Affiliation: National Institute of Informatics, Tokyo, Japan  
E-mail: mahito@nii.ac.jp
