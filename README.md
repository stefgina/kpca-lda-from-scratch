# Dimensionality Reduction MNIST

```python
Author -> Stefanos Ginargyros
```

## Dependencies

In order to run the script you will have to install the following dependencies:

```
pip install scikit-learn
pip install numpy
pip install scipy
pip install matplotlib
```

## Dataset

The dataset for this problem is the famous MNIST. This particular version of MNIST is from Yann Lecun, and it can be obtained from 
http://yann.lecun.com/exdb/mnist/ or you can just unzip it from my repo. In order to load the data from the official compressed form, I wrote 
a script which translates the first 16 bytes of the idx3 file into size and dimension data, and then loads the rest (pixels) into np arrays of 
the specific dimensionality. Even though the whole dataset was 70k images and labels, I utilized only the 25k of them, in order for the KPCA to 
be inside the memory bounds of my system. (15k training – 10 testing).

<img src="https://github.com/stefgina/kpca-lda-mnist/blob/main/mnist.png" width=400>


## KPCA (from scratch)

I wrote the kpca function from scratch, using only numpy and some linear algebra from scipy. Although you can still replace it, with 
sklearn's built-in kpca for even better results. I did it in order to understand the maths, but their version is more optimized.

```kpca()``` does the following:
- Calculates the kernel matrix/ centering
- Extracts eigenvectors and eigenvalues of K
- Returns the projected data/ eigenvectors based on the number of principal components.

```project_kpca()``` does:
- Projecting the test into the principal component space made by the train-set. Otherwise if you kpca on both is cheating (cherry-picking).

For this particular problem I shaped kpca's output to 300, meaning that ```components=300```. This way we hold more than 90% of 
the initial information. Gamma was set 1, meaning that ```gamma=1``` as a parameter in our ```kpca(X, components=300, gamma=1)```. So even 
though we lost more than half the features that we initially got (from 784 to 300), it turns out that is easier for our classifier to handle 
the data this way achieving even greater results.

## LDA (sklearn's)

As for this part, I used the prebuilt ```lda()``` from sklearn. Empirically is better to pass inside the LDA the output of KPCA. So from 784 
initial features we have 300, and after LDA we 'll have just 9. Keep in mind that LDA needs supervision, in contrast with KPCA. The final 
dimension of the data is 15000 x 9.

## Classification/ Results

Custom KPCA+LDA took 3 minutes: 
- Nearest Neighbor achieved 0.84 accuracy on test, which is +0.03 better than without any dimensionality reduction at all.
- K-NN achieved 0.87 accuracy on test, which is +0.01 better than without any dimensionality reduction at all.

Sklearns KPCA+LDA took 2.40 minutes:
- K-NN (n=5) under prebuild libraries achieved 0.91 accuracy for the 15k trainset which is +0.04 better than my KPCA, and +0.05 better than 
without any KPCA or LDA at all.
- SVM achieved 0.97 with the prebuilt libraries, which is +0.6 better from the run without any kpca+lda.  (full dataset NOT just 15k for training)
- Nearest Centroid under prebuilt libraries achieved 0.88 for the 15k trainset which is +0.04 better than my KPCA, and +0.07 better than without any KPCA or LDA at all.





