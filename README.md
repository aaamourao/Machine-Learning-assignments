# Machine Learning
> Assignments' solutions for ML discipline on DCC/UFMG
>
> @aaamourao
>
> mourao.aaa@gmail.com
>
> encrypted mail: adrianomourao@protonmail.com

## PA1
### Spec

*Full document available on [PA1.pdf](https://github.com/aaamourao/Machine-Learning-assignments/blob/master/PA1/PA1.pdf)*

## Goal
Implement Perceptron and SVM via Pegasos for detecting spam mail. The dataset was provided by *SpamAssassin Public* and it's
already preprocessed: lower casing, removal of HTML tags, normalization of URLs, e-mail addresses and numbers.

## Code, analysis and result

### Perceptron approach
Source code of [Perceptron class](https://github.com/aaamourao/Machine-Learning-assignments/blob/master/PA1/src/perceptron.py): Can be used on general applications!

* Training code example: [SpamAssassin](https://github.com/aaamourao/Machine-Learning-assignments/blob/master/PA1/src/train.py)
* Target error on training **[overfitting]**: `0.00%`
* Error on validation data: `3.6%`

![Results of SpamAssassin](https://github.com/aaamourao/Machine-Learning-assignments/blob/master/PA1/src/training-data/training.png)

For more details check [the project wiki page](https://github.com/aaamourao/Machine-Learning-assignments/blob.wiki)!
