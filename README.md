# Robust Suicide Risk Assessment on Social Media via Deep Adversarial Learning

This codebase contains the python scripts for ASHA, the base model for Robust Suicide Risk Assessment on Social Media via Deep Adversarial Learning.

Accepted at Journal of the American Medical Informatics Association ([paper coming soon!](#))

## Environment & Installation Steps

Python 3.8 & Pytorch 1.7

## Run

Execute the following steps in the same environment:

```bash
cd asha-jamia & python main.py
```

## Command Line Arguments

To run different variants of ASHA, perform ablation or tune hyperparameters, the following command-line arguments may be used:

```
usage: main.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--num-runs NUM_RUNS]
               [--early-stop EARLY_STOP] [--hidden-dim HIDDEN_DIM] [--embed-dim EMBED_DIM]
               [--num-layers NUM_LAYERS] [--dropout DROPOUT] [--learning-rate LEARNING_RATE]
               [--epsilon EPSILON] [--scale SCALE] [--data-dir DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size (default: 8)
  --epochs EPOCHS       number of epochs (default: 50)
  --num-runs NUM_RUNS   number of runs (default: 50)
  --early-stop EARLY_STOP
                        early stop limit (default: 10)
  --hidden-dim HIDDEN_DIM
                        hidden dimensions (default: 256)
  --embed-dim EMBED_DIM
                        embedding dimensions (default: 768)
  --num-layers NUM_LAYERS
                        number of layers (default: 1)
  --dropout DROPOUT     dropout probablity (default: 0.4)
  --learning-rate LEARNING_RATE
                        learning rate (default: 0.01)
  --epsilon EPSILON     value of epsilon (default: 0.1)
  --scale SCALE         scale factor alpha (default: 1.8)
  --data-dir DATA_DIR   directory for data (default: )
```

## Dataset

We use the dataset released by [1] that consists of reddit posts of 500 users across 9 mental health and suicide related subreddits.

https://github.com/AmanuelF/Suicide-Risk-Assessment-using-Reddit

Processed dataset format should be a DataFrame as a .pkl file having the following columns:

1. label : 0, 1, ... 4 denoting the risk level of the user.
2. enc : list of lists consisting of 768-dimensional encoding for each post.

## Some code was forked from the following repositories:
 
 - [sismo-wsdm](https://github.com/midas-research/sismo-wsdm)
 - [STATENet_Time_Aware_Suicide_Assessment](https://github.com/midas-research/STATENet_Time_Aware_Suicide_Assessment)

## Ethical Considerations

We work within the purview of acceptable privacy practices suggested by [2] and considerations discussed by [3] to avoid coercion and intrusive treatment.
For the dataset [1] used in this research, the original Reddit data is publicly available.
Our work focuses on developing a neural model for screening users and does not make any diagnostic claims related to suicide.
We study Reddit posts in a purely observational capacity, and do not intervene with user experience.
The assessments made by SISMO are sensitive and should be shared selectively and subject to IRB approval to avoid misuse like Samaritan’s Radar [4].

## Cite

If our work was helpful in your research, please kindly cite this work:

```
@inproceedings{sawhney2021ordinal,
    author={Sawhney, Ramit  and
            Joshi, Harshit  and
            Gandhi, Saumya  and
            Shah, Rajiv Ratn},
    title = {Towards Ordinal Suicide Ideation Detection Social Media},
    year = {2021},
    month=mar,
    booktitle = {Proceedings of 14th ACM International Conference On Web Search And Data Mining},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    keywords = {social media, suicide ideation, ordinal regression, reddit},
    location = {Virtual Event, Israel},
    series = {WSDM '21}
}
```

## References

[1] Gaur, M., Alambo, A., Sain, J. P., Kursuncu, U., Thirunarayan, K., Kavuluru, R., ... & Pathak, J. (2019, May). Knowledge-aware assessment of severity of suicide risk for early intervention. In The World Wide Web Conference (pp. 514-525).

[2] Chancellor, S., Birnbaum, M. L., Caine, E. D., Silenzio, V. M., & De Choudhury, M. (2019, January). A taxonomy of ethical tensions in inferring mental health states from social media. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 79-88).

[3] Fiesler, C., & Proferes, N. (2018). “Participant” perceptions of Twitter research ethics. Social Media+ Society, 4(1), 2056305118763366.

[4] Hsin, H., Torous, J., & Roberts, L. (2016). An adjuvant role for mobile health in psychiatry. JAMA psychiatry, 73(2), 103-104.
