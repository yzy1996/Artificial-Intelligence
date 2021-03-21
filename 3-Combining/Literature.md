**[`evolutionary-stochastic-gradient-descent-for-optimization-of-deep-neural-networks`]**

**[`2018`]** **[`NIPS`]** **[[:memo:]()]** **[[:octocat:]()]**

<details><summary>Click to expand</summary><p>


**Abstract:**

> They propose a population-based Evolutionary Stochastic Gradient Descent (ESGD) framework for optimizing deep neural networks.
> 
>This ESGD framwork can be applied on  speech recognition, image recognition and language modeling, using networks with a variety of deep architectures.

**The methods it used:** 

- [x] Some optimizer including SGD, ADAM (ref my [blog](https://blog.csdn.net/yzy_1996/article/details/84618536)).
- [x] Some hyper-parameters including learning rate, momentum.

**main work**

> Their method has two main steps, each species evolves independently in the SGD step and then interacts with each other in the EA step. 

**Its contribution:**

> ESGD not only integrates EA and SGD as complementary optimization strategies but also makes use of complementary optimizers under this coevolution mechanism. And they showed the effectiveness of ESGD across some classical dataset .

**My Comments:**

> The step of SGD is easy to understand. How do these parameters evolve further? They just use different hyper-parameters to evolve the individual parameters and finally select **m-elitist** offspring population. The idea is clear and simple, and in fact consistent with mine.

**Valuable Sentences**

>Evolutionary algorithms are population-based so computation is intrinsically parallel.

</p></details>

---