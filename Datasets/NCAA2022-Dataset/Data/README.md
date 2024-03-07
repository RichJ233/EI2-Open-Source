# NCAA2022-dataset

## Introduction

Recently, energy prediction, weather prediction, market prediction, pandemic spreading prediction, and other time-series prediction research areas have gained increasing attentions. For those typical problems, researchers have proposed various approaches based on machine learning and neural network. However, there is still alack of a comprehensive dataset for a fair evaluation. As a result, NCAA2022 datasets, including 16 prediction problems, are proposed to cover most characteristics of prediction model. 

According to the frequency domain characteristics, four typical datasets with non-stationary, periodic, impulsive, and chaotic characteristics are transformed into four types of problems, i.e., low-pass, high-pass, band-pass, and band-stop categories. The prediction results based on NCAA2022 time-series datasets could reflect the performance of different frequency domains, and identify which type of problems can be solved effectively for a certain testing method. The algorithm robustness of complex tasks could be described by the comprehensive performance on 16 prediction problems.

## Includes

    ├─Answer
    │   │
    │   └─Dataset 1-16
    │
    ├─Problem
    │   │
    │   └─Dataset 1-16
    │
    └─README.md

## Citation

If you find our work useful in your research, please consider citing: 
```
@article{Wu2023time,
  title={Time-series benchmarks based on frequency features for fair comparative evaluation},
  author={Wu, Zhou and Jiang, Ruiqi},
  journal={Neural Computing and Applications},
  year={2023},
  publisher={Springer}
 doi={https://doi.org/10.1007/s00521-023-08562-5}
}
```

## Acknowledgements

- It's delighted to be used in academic research, and please follow the [MIT license](./LICENSE). If any business requires, please contact zhouwu@cqu.edu.cn or jiang_ruiqi@cqu.edu.cn.