**iGAN** proposes to reconstruct and edit a real image using GANs. The method first projects a real photo onto a latent vector using a hybrid method of encoder-based initialization and per-image optimization. It then modifies the latent vector using various editing tools such as color, sketch, and warping brushes and generates the final image accordingly. Later, **Neural Photo Editing** proposes to edit a face photo using **VAE-GANs**.

Recently, **GANPaint** [8] proposes to change the semantics of an input image by first projecting an image into GANs, then fine-tuning the GANs to reproducing the details, and finally modifying the intermediate activations based on user [inputs](Gan dissection: Visualizing and understanding generative adversarial networks). 





## iGAN(aka. interactive GAN)

Generative Visual Manipulation on the Natural Image Manifold

code:https://github.com/junyanz/iGAN





GAN Dissection: Visualizing and Understanding Generative Adversarial Networks

[Home page](https://gandissect.csail.mit.edu/)

[tutorial](https://medium.com/@xiaosean5408/gan-dissection%E7%B0%A1%E4%BB%8B-visualizing-and-understanding-generative-adversarial-networks-37125f07d1cd)





Transforming and Projecting Images into Class-conditional Generative Networks





