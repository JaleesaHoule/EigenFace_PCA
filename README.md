# BasicPCA
implementation of EigenFace theory for pattern recognition course

## PCA theory and procedures

PCA is a method of dimensionality reduction. The goal of PCA is to find a basis of eigenvectors (i.e., principle components) which describes the data. PCA is useful in that it de-correlates the data and preserves original variances of the data. This allows one to reduce the number of dimensions of their data for classification while still preserving a large amount of information/variance in the data. To perform PCA on a database of images, the database of images must first be shaped into an array of shape $(H\text{x}W, M)$, where $M$ is the total number of images, $H$ is the height of each image, and $W$ is the width of each image. Given a database of images, the sample mean of the images can be computed as

```math
\begin{equation}
    \mathbf{\bar{x}} = \frac{1}{M}\sum\limits_{i=1}^{M}\mathbf{x_i}.
\end{equation}
```math

Once the sample mean is computed, it is subtracted from each image ($\Phi_i = \mathbf{x} - \mathbf{\bar{x}}$) so that the data is centered at zero. The resulting matrix of images is then $\mathbf{A}=[\Phi_1 \Phi_2 ... \Phi_M]$. Once the data is centered, we can compute the sample covariance matrix such that
```math
\begin{equation}
\begin{aligned} 
    \mathbf{\Sigma_x} &= \frac{1}{M}\sum_{i = 1}^M(\mathbf{x} - \mathbf{\Bar{x}})(\mathbf{x} - \mathbf{\Bar{x}})^\top \nonumber\\
        &= \frac{1}{M}\sum_{i = 1}^{M}\mathbf{\Phi}_i\mathbf{\Phi}_i^\top \\
            &= \frac{1}{M}\mathbf{A}\mathbf{A}^\top. \\
\end{aligned}
\end{equation}
```
Next, we can compute the eigenvectors and eigenvalues of $\Sigma_x$ using the formula $\mathbf{\Sigma_x}\mathbf{u_i} = \lambda_i\mathbf{u}_i$. In practice, this is computationally expensive, since $\mathbf{AA}^T$ is of shape ($H\text{x}W, H\text{x}W$). Therefore, instead of using the matrix $\mathbf{AA}^T$, we can instead use the matrix $\mathbf{A}^T\mathbf{A}$, which is of size ($M\text{x}M$) and is generally significantly smaller than $\mathbf{AA}^T$ ($M<<H\text{x}W$). This "trick" can be performed due to the relationship between the two matrices, where if we assume that $(\mathbf{AA}^T)\mathbf{v}_i=\mathbf{u}_i\mathbf{v}_i$ and $(\mathbf{A}^T\mathbf{A})\mathbf{v}_i=\mathbf{
\mu_i}\mathbf{v}_i$, then, multiplying both sides by $\mathbf{A}$ gives

```math
\begin{equation}
\begin{aligned}
    \mathbf{A}(\mathbf{A}^\top\mathbf{A}) \mathbf{v}_i &= \mathbf{A}\mu_i\mathbf{v}_i  \text{ , i.e., } \\ 
        (\mathbf{A}\mathbf{A}^\top)\mathbf{A} \mathbf{v}_i &= \mu_i(\mathbf{A}\mathbf{v}_i). \\ 
\end{aligned}
\end{equation}
```
If we also assume that $(\mathbf{AA}^T)\mathbf{v}_i =  \mathbf{\lambda}_i\mathbf{u}_i$, then this manipulation shows that

```math
\begin{equation}
\begin{aligned}    
    \mathbf{\lambda_i} &= \mathbf{\mu_i} \\
    \mathbf{u_i} &= \mathbf{Av}_i. \\
    \label{AA_transform}
\end{aligned}
\end{equation}
```

Since $\mathbf{\lambda_i} = \mathbf{\mu_i}$, it can be shown that the eigenvalues of $\mathbf{A}^T\mathbf{A}$ are the top $M$ eigenvalues of $\mathbf{AA}^T$. Additionally, the top $M$ eigenvectors of $\mathbf{AA}^T$ can be found using the relationship $\mathbf{u_i} = \mathbf{Av}_i $. This relationships allows us to calculate the desired eigenvalues and eigenvectors of $\mathbf{AA}^T$ by using the much smaller matrix $\mathbf{A}^T\mathbf{A}$. 

Once the eigenvectors of $\mathbf{A}^T\mathbf{A}$ are found and transformed to be the eigenvectors of $\mathbf{AA}^T$ using Equation \ref{AA_transform}, they are normalized to unit length (i.e., so that $||\mathbf{u_i}=1||$). These unit length eigenvectors form a basis, and each $\mathbf{u_i}$ can be transformed back into the image space using the relationship
```math
\begin{equation}
    \mathbf{w_{ij} = }\frac{255 \mathbf{(u_{ij} - u_{\text{min}})}}{\mathbf{u_{\text{max }- u_{\text{min}}}}}
\end{equation}
```
where $\mathbf{w_{ij}}$ represents integer values between $[0,255]$. In projecting these basis vectors back into the image space, we see that each eigenvector looks like a "ghost face", which is why they are typically referred to as eigenfaces. We can reconstruct any image $x$ in the image gallery by using the equation
```math
\begin{equation}
    \mathbf{x - \bar{x}} \approx \sum\limits_{i=1}^{M}\mathbf{y_iu_i}

\end{equation}
```
where $\mathbf{y_i}$ is an eigen-coefficient which describes the weight of each eigenvector needed to reconstruct $\textbf{x}$. To reduce dimensionality, we can take the top $K$ largest eigenvalues/eigenfaces ($K<<M$) and approximate the image $\hat{\mathbf{x}}$ as 
```math
\begin{equation}
    \mathbf{\hat{x} - \bar{x}} \approx \sum\limits_{i=1}^{K}\mathbf{y_iu_i}.
\end{equation}
```
This reduction allows us to preserve some chosen threshold of information within the data, while drastically reducing dimensionality and computational time. \\


## Choosing a threshold for $K$

Since each eigenvalue found from $\Sigma_x$ corresponds to variance for a feature, the top $K$ eigenvalues values can be thought of as corresponding to some percentage of information of the data. To choose a value of $K$ which preserves a desired amount of information/variance from the data, we can use the relationship
```math
\begin{equation}
    \frac{\sum\limits_{i=1}^{K}\mathbf{\lambda_i}.}{\sum\limits_{i=1}^{M}\mathbf{\lambda_i}} > T_r
\end{equation}
```
where $T_r$ is a value between 0 and 1 (i.e., to preserve 90\% of the data, we would set $T_r=0.9$ and solve for $K$. The resulting $K$ eigenvalues and their corresponding eigenvectors would retain 90\% of the information from the original data. We could use the approximate reconstruction error ($|| \mathbf{x}- \mathbf{\hat{x}}||$) to determine how close the estimation ($\mathbf{\hat{x}}$) is compared to the original image (\textbf{x}).\\

## Face recognition using the Mahalanobis distance
In addition to reconstructing images within the training data, we can also see how well new test images are recognized. During the training phase, we can compute the eigen-coefficients for each training image such that 
```math
\begin{equation}
   \mathbf{ \Omega_i} = \begin{bmatrix}
        y_1 \\
        y_2 \\
        \cdot\\
        \cdot\\
        y_K \\  
    \end{bmatrix}.
\end{equation}
```
Then, we can do the same for an unknown image $\Phi$, by first subtracting the mean face ($\Phi = \mathbf{x - \hat{x}}$) and then projecting the unknown face into the eigenspace so that 
```math
\begin{equation}
    y_i = \Phi^T \mathbf{u_i} \text{, for }i = 1,2,..., K.
\end{equation}
``
We can then compare the eigen-coefficients of the unknown face ($\Omega$) with the eigen-coefficients of the known faces ($\Omega_i$) and find the closest match such that 
```math
\begin{equation}
    p = \text{arg min}_i||\Omega - \Omega_i || \text{, for }i = 1,2,..., M.
\end{equation}
```
For this assignment, we will compute this using the Mahalanobis distance, where
```math
\begin{equation}
    e_r = \text{min}_i ||\Omega - \Omega_i ||  = \text{min}_i \sum\limits_{j=1}^{K}\frac{1}{\lambda_j}(y_j-y^i_j)^2.
\end{equation}
```
Turk & Pentland (1991), refer to $e_r$ as the difference in face space (difs). By computing the difs between an unknown testing image and every training image, we can assign the training image with the smallest distance as the recognized face of the unknown image. In practice this measurement is not perfect, therefore some faces may be recognized as the wrong person. To see how well the $e_r$ measurement does at recognizing a novel face, we will take the top $r$ images corresponding to the $r$ smallest distances between the unknown faces and the faces in the training image database. Then, we will plot a Cumulative Match Characteristic (CMC) curve (\cite{phillips2000feret}) by varying $r$ and counting how many test faces are correctly recognized within the $r$ top matches. 
