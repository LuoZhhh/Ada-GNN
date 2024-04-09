# Note

This repository includes the implementation for our WSDM 2022 paper: ***[Ada-GNN: Adapting to Local Patterns for Improving Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3488560.3498460).***

## Environments

Python 3.7.6

Packages:

```
dgl_cu102==0.9.1.post1
learn2learn==0.1.5
matplotlib==3.3.4
torch==1.12.1+cu102
numpy==1.19.2
dgl==2.1.0
scikit_learn==1.4.1.post1
```

Run the following code to install all required packages.

```
> pip install -r requirements.txt
```

## Datasets & Processed files

- The datasets include Amazon and Arxiv, please download the dataset from the ***[google drive](https://drive.google.com/file/d/14zZN4CM8Am1ipJYQ9gcjtlcBmougODdS/view?usp=sharing)*** and save as the `dataset/` folder.
- Remember to create `saved_model/` folder for saving the model. The full repository should be as follows:

  ```
  .
  ├── dataloader.py
  ├── dataset
  ├── main.py
  ├── model.py
  ├── motivation.py
  ├── partition.py
  ├── readme.md
  ├── requirements.txt
  ├── saved_model
  └── utils.py
  ```

## About MAML
The implementation of MAML is inspired from ***[this repository](https://github.com/learnables/learn2learn/tree/master)***.

## Run the code
For instance, to run Ada-GraphSAGE on Arxiv dataset, please use this command:
```
python main.py --dataset arxiv --model SAGE --num-parts 3 --num-steps 5
```
To run Ada-GraphSAGE-fair, please use this command:
```
python main.py --dataset arxiv --model SAGE --num-parts 3 --num-steps 5 --fairness
```

## BibTeX

If you like our work and use the model for your research, please cite our work as follows.

```bibtex
@inproceedings{luo2022Ada-GNN,
  author       = {Zihan Luo and
                  Jianxun Lian and
                  Hong Huang and
                  Hai Jin and
                  Xing Xie},
  editor       = {K. Selcuk Candan and
                  Huan Liu and
                  Leman Akoglu and
                  Xin Luna Dong and
                  Jiliang Tang},
  title        = {Ada-GNN: Adapting to Local Patterns for Improving Graph Neural Networks},
  booktitle    = {{WSDM} '22: The Fifteenth {ACM} International Conference on Web Search
                  and Data Mining, Virtual Event / Tempe, AZ, USA, February 21 - 25,
                  2022},
  pages        = {638--647},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3488560.3498460},
  doi          = {10.1145/3488560.3498460},
  timestamp    = {Sat, 30 Sep 2023 09:59:27 +0200},
  biburl       = {https://dblp.org/rec/conf/wsdm/LuoL00022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
``` 