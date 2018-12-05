# Triangle Counting using SIMT parallel set intersection on GPU
Algorithm features:
* Redirect edges based on degree order to reduce the computation complexity
* Use CSR(Compressed Sparse Row) format to reduce the memory usage on GPU
* Use SIMT to parallel the sorted lists intersections on GPU

## Problem Definition
The problem is from the competition 2018 CCF BDCI in China, and we are very lucky to win the **first prize**.

[基于GPU服务器的图数据三角形计数算法设计与性能优化](https://www.datafountain.cn/competitions/321/details/rule)

## Dataset

|      Dataset      | Vertices  |   Edges    |  Triangles   |
| :---------------: | :-------: | :--------: | :----------: |
|  twitter_rv.bin   | 61578415  | 1468365182 | 34824916864  |
| s26.kron.edgelist | 67108861  | 1073741824 | 49167172995  |
| s27.kron.edgelist | 134217725 | 2147483648 | 106869298996 |

[Download datasets](https://pan.baidu.com/s/1zCYCZPAw_jz346YMvWAGfw)

## How it works

Google Drive: [【BDCI 2018】欧拉的核弹PPT](https://drive.google.com/open?id=13FXcvfi1H63FvJwicCBlBTdkqvtAkKxE)

Baidu Netdisk: [【BDCI 2018】欧拉的核弹PPT](https://pan.baidu.com/s/1EYZ6EzaAM0yqu0Be8wnF_A)

## How to Use
Build the project with:
```shell
cd src
make all
```
If you wanna use gpu version,  run
```
./tricount -f {{your dataset path}}
```
or if you wanna try the old cpu version
```
./tricount_cpu -f {{your dataset path}}
```

## License
Apache License 2.0
