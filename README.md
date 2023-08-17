# K-NN Machine learning training

This project uses the [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) dataset. 

To get started, create a `data` directory in the root directory of this project, download the dataset and unpack it inside `data`.
```bash
$ mkdir data ; cd data
$ wget http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
$ tar -xf PKLot.tar.gz
```

You should also install the **opencv and boost c++** libraries.

## Usage
Before compiling, you should open the `main.cpp` file and set `THREAD_NUM` global variable to the number of threads to be used in the KNN testing phase.  

Then, compile it with CMake and make. Finally, run the `main` binary file.
```bash
$ cmake .
$ make
$ ./main
```
