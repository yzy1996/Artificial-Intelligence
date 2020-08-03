首先不谈公式，为了让你有个最直观的认识



 GANs involve training a generator and discriminator model in an adversarial game







首先从判别器的损失函数开始说起，对于判别器，他需要对 原真实数据 和 新生成数据 分别打分，记作 $D(x)$ $D(G(z))$ ，真实标签分别为 $1 \quad 0$

对他们求交叉熵，因为是二分类问题，所以使用的是 binary cross entropy
$$
BCE(D(x)) = -[1*\log(D(x)) + 0*\log(1-D(x))] = -\log(D(x))
$$

$$
BCE(D(G(z))) = -[0*\log(D(G(z))) + 1*\log(1-D(G(z)))] = -\log(1-D(G(z)))
$$

因此判别器的损失函数是上述两者的和，同时对数据进行了采样取均值处理
$$
\begin{align}
J^D 
& = \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[BCE(D(x))] + \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[BCE(D(G(z)))] \nonumber\\
& = - \left[\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]\right]
\end{align}
$$
此时是需要最小化 $J^D$，去掉负号后，变成了最大化 $-J^D$，到这里我们就得出了原表达式的一部分
$$
\max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$


现在再来看生成器的损失函数，与上面类似，但现在只有判别器给 新生成数据 打分，记作 $D(G(z))$，真实标签为 1。注意这里是与上面反着的，判别器为了让自己给这个假数据打0分，而生成器为了骗判别器给这个假数据打1分。交叉熵表示为
$$
BCE(D(G(z)) = -[1*\log(D(G(z))) + 0*\log(1-D(G(z)))] = -\log(D(G(z)))
$$
于是

但现在
$$
-\log(D(G(z))) \rightarrow \log (1-D(G(\boldsymbol{z})))
$$
正确标签找最小到错误标签找最大，一个是看正确率（越小越好），一个是看错误率（越大越好），
$$
\min J^G = \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$


所以我们得到了最终的表达式：
$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$
