![img](https://img-blog.csdnimg.cn/20190930155217606.png)

m ![img](https://img-blog.csdnimg.cn/20190930155245918.png) where

- **tensor** – an n-dimensional torch.Tensor
- **a** – the negative slope of the rectifier used after this layer (0 for ReLU by default)
- **mode** – either `'fan_in'` (default) or `'fan_out'`. Choosing `'fan_in'` preserves the magnitude of the variance of the weights in the forward pass. Choosing `'fan_out'` preserves the magnitudes in the backwards pass.
- **nonlinearity** – the non-linear function (nn.functional name), recommended to use only with `'relu'` or `'leaky_relu'` (default).