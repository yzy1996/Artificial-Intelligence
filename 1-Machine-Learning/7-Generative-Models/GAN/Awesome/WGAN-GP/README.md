# WGAN-GP

[论文](https://arxiv.org/abs/1704.00028) [代码](https://github.com/igul222/improved_wgan_training)



```python
def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)

def gradient_penalty(f, real_data, fake_data):

    shape = [real_data.shape[0]] + [1] * (real_data.shape.ndims - 1)
    alpha = tf.random.uniform(shape=shape)
    
    interpolate = real_data + alpha * (fake_data - real_data)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolate)
        interpolate_output = f(interpolate)
        
    gradient = tape.gradient(interpolate_output, interpolate)
    norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)

    return tf.reduce_mean((norm - 1.) ** 2)

discriminator_loss_fn() += 10 * gradient_penalty()
```


在WGAN中，我们用裁切的方式实现利普希茨约束，但裁切的参数 $c$ 
$$
w \leftarrow \operatorname{clip}(w,-c, c)
$$
如果选择不恰当，会导致梯度消失，很难收敛。



所以现在我们不使用权重裁切，而是给权重加一个惩罚，同样能实现利普希茨连续性。

经过证明，再真实数据和生成数据之间的插值点处应该满足梯度的范数为1

所以
$$
L=\underbrace{\underset{\tilde{\boldsymbol{x}} \sim \mathbb{P}_{g}}{\mathbb{E}}[D(\tilde{\boldsymbol{x}})]-\underset{\boldsymbol{x} \sim \mathbb{P}_{r}}{\mathbb{E}}[D(\boldsymbol{x})]}_{\text {Original critic loss }}+\underbrace{\lambda \underset{\boldsymbol{\Phi} \sim \mathbb{P}_{\boldsymbol{\omega}}}{\mathbb{E}}\left[\left(\left\|\nabla_{\hat{\boldsymbol{x}}} D(\hat{\boldsymbol{x}})\right\|_{2}-1\right)^{2}\right]}_{\text {Our gradicnt penalty }}
$$
where $\hat{x}$ sampled from fake $\tilde{x}$ and real $x$ 
$$
\hat{\boldsymbol{x}}=t \tilde{\boldsymbol{x}}+(1-t) \boldsymbol{x} \text { with } 0 \leq t \leq 1
$$
and $\lambda$ is set to 10.



WGAN-GP中，对判别器网络不再使用Batch normalization，因为会影响惩罚的效果



![Image for post](https://miro.medium.com/max/2054/1*yYvwVzRnlVmRFCh7-JOASw.png)





https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490