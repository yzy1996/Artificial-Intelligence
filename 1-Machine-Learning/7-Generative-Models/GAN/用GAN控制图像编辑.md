首先有一个重建的工作



## Image Reconstruction



We want to map from feature space to latent space

given a sample ,we want to get a corresponding z



separate train an encoder, input a image and output a z vector

To verify z is correct, feed it into G and then check whether the generated image is similar to the input image



design E same with D, but the last layer output the lenth of z





Adversarially Learned Inference

Autoencoding beyond pixels using a learned similarity metric





After we have a specific class of cluster image z

if we want to find a contribute

we can substract the average z vector of all images that do not have the attribute from the average z vector of all images that have it



