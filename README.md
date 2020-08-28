# OpenMP-Kmeans
To perform k-means clustering and parallelizing it through OpenMP.  
 Value of k is asked from user.  
 iris.data is given as a dataset. Read the data from file and ignore the last column of each row in dataset.

To compile:
```
gcc -fopenmp -o k_means k_means.c
```
where k_means is the name of the object file.

To run:
```
./k_means
```
Number of threads will be asked at the starting when the code is run. You can give any positive integer value in that. But generally 3-4 threads give best results.
