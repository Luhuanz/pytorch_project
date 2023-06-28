# Improved GAN Training

####  **Feature Matching**

Feature matching suggests to optimize the discriminator to inspect whether the generator’s output matches expected statistics of the real samples. In such a scenario, the new loss function is defined as$\left|\mathbb{E}_{x \sim p_r} f(x)-\mathbb{E}_{z \sim p_z(z)} f(G(z))\right|_2^2$,where $f(x)$ can be any computation of statistics of features, such as mean or median.

#### **Minibatch Discrimination**

With minibatch discrimination, the discriminator is able to digest the relationship between training data points in one batch, instead of processing each point independently.

####  **Historical Averaging**



####  **One-sided Label Smoothing**



#### **Virtual Batch Normalization** (VBN)



####  **Adding Noises**.



### 