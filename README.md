# GLCIC-pytorch

## About this repository

Here, we provide a pytorch implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

![network](https://i.imgur.com/wOnxWNc.png "Network")

## Algorithm

![algorithm](https://i.imgur.com/pdVz4Tf.png "Algorithm")

* Mc: completion region mask (it takes the value 1 inside regions to be completed and 0 elsewhere)
* Md: random mask

The training is split into thress phases.    
* PHASE1: the completion network is trained with Equation 2 for Tc iterations.
* PHASE2: only the context discriminators are trained with Equation 3 for Td iterations. At this time, the completion network is frozen.
* PHASE3: Both the completion network and the context discriminators are trained jointly until the end of the training.

**Equation 2**  
![eq2](https://i.imgur.com/zRI5YgA.png)

**Equation 3**  
![eq3](https://i.imgur.com/e4AhoUg.png)

**Equation 4**  
![eq4](https://i.imgur.com/40IwojH.png)

## Notes

* Because the completion network is a FCNN, input images don't have to be resized.
* completion regions of a training image are filled with the mean pixel value of all the training images.
