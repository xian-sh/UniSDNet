The Response to Reviewer Z5of (Part 1/3)

Thank you very much for your recognition of our contributions to NLVG and SLVG tasks. Your constructive comments are exceedingly helpful for us to improve our paper. 

> **Q1: The inspiration from human visual perception biology is not very motivating. Specifically, it is hard to see why a MLP with residual connection is the way to achieve the 'global broadcast communication' of the brain. Either bridge the gap or Simply drop the bio-inspiration and go straight into the technical method.**

**A1:**
Thank you very much for your attention to the design motivation of static networks (MLP with residual connection, ResMLP). Here, we clarify the motivation to bridge the gap between the biological theory 'global broadcast communication' and the technical design of static network ResMLP. 
Sincerely, our work has been strongly inspired by the latest theories of human visual perception [R14, R15]. The **global neuronal workspace (GNW)** theory mentioned in [R15] is the foundation of conscious work in visual understanding [R15, R16, R17], it can achieve the fusion process of multi-source information in the early stage of visual event recognition, that is, the **"global broadcast communication"** of the brain. GNW theory indicates that cognitive consciousness has the function of **strengthening perceptual conditions** and **delaying conditioned reflexes** [R15, R18], enabling the human brain to understand new conditioned reflexes brought about by perceptual information updates [R15, R16].

On the premise of the above theory, this inspires us to utilize residual network structure to update our understanding of video sequences in the delaying conditioned reflex states by aggregating the perceptual information. **Multilayer perceptron (MLP)** is proven to be effective and has strong interpretability in modeling multi-level cognitive processing [R19], thus we use MLP to aggregate the perceptual information. The visual consciousness process is modeled as follows: $x^{n+1} = \text{ResMLP}(x^n) = x^n+\text{MLP}(x^n),$ where $x^n$ represents the perception condition in the current state, $x^{n+1}$ denotes the reflection result, and $\text{ResMLP}(x^n)$ can enhance the information of the perception condition $x^n$.

Technically, in this work, we adopt a unified framework of static (ResMLP) and dynamic (graph filtering) structures for video grounding. In the early stage, this global broadcast mode of ResMLP first perceives all multimodal information, while the latter graph filter purifies key visual information. 
This pre-broadcast learning mode of ResMLP is very necessary as a global biological perception mechanism and has also verified its effectiveness via extensive experiments, as discussed in the paper. We will add the explanation on the rationality of our static network design from a **technical perspective** and incorporate your suggestion into the paper writing by considering both motivation and technical implements for better clarification. 

**References:**

[R14] Barbosa et al., Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory. Nature neuroscience, 23(8), pp. 1016-1024, 2020. 

[R15] Volzhenin et al., Multilevel development of cognitive abilities in an artificial neural network. Proceedings of the National Academy of Sciences, 119(39), pp. e2201304119, 2022.

[R16] Cleeremans et al., Consciousness matters: Phenomenal experience has functional value. Neuroscience of consciousness. 2022, pp. niac007, 2022.

[R17] Richards et al., A deep learning framework for neuroscience. Nature neuroscience, 22, pp. 1761-1770, 2019.

[R18] Grover et al., Differential mechanisms underlie trace and delay conditioning in Drosophila. Nature, 603, pp. 302-308, 2022. 

[R19] Chavlis et al., Drawing inspiration from biological dendrites to empower artificial neural networks. Current opinion in neurobiology. 70, pp. 1-10, 2021. 


The Response to Reviewer Z5of (Part 2/3)

> **Q2: When expanding a single gaussian kernel to multi-kernel Gaussian, it seems that only the bias z is sweeping? Have you tried different $\gamma$?**

**A2:** Thank you for the valuable feedback. 
Considering to better model the continuity and complexity of visual perception transmission, **a single-kernel Gaussian filter** is insufficient to reflect the comprehensive degree of event associations in the video [R14]. We adopt **multi-kernel Gaussian filters** as a rich **Filter-generating-networks** architecture to extend the scalar $d_{ij}$ to high-dimensional space to achieve the goal. 
In this work, we employ the multi-kernel Gaussian $\phi_k(x)=exp(-\gamma(x-z_k)^2),  k \in [1, h]$, and there are three variables ($z_k,h,\gamma$): different bias $\{z_k\}$ for total $h$ Gaussian kernels and a Gaussian coefficient $\gamma$, where $z_k$ is a bias to avoid a plateau at the beginning of training due to the highly correlated Gaussian filters. 
To meet the constraint of nonlinear correlated Gaussian kernels, we randomly set biases $z_k$ at equal intervals (e.g., 0.1 or 0.2) starting from 0.0, sweep the value of $h$ from 25 to 200 (see the ablation experiment Table 9 in Appendix and the best setting is $h=50$) and set the global range of $\{z_k\}$ values to $[0, 5]$ in our experiments.

As per your valuable suggestion, we add new experiments about different $\gamma$ in **Table R4**. Gaussian coefficient $\gamma$ reflects the amplitude of Gaussian kernel function that controls the gradient descent speed of the function value.
It can be find that from **Table R4**, when $\gamma=25.0$, our model achieves the best performance (in bold font of **Table R4**). We also list the average and standard deviation of the five experimental results of $\gamma=\{5.0, 10.0, 25.0, 50.0, 70.0\}$. 
We select $\gamma=10.0$ as the empirical setting as its result (with *italic* in **Table R4**) is closest to the average *avg.*. To summarize, in our experiments, the final settings of variables  ($h,\gamma$) are set to 50 and 10.0, and $\{z_k\}$ is set at an equal interval of 0.1. 
We will add this new ablation and more explanation to the paper.


**Table R4: The ablation study about different Gaussian coefficients $\gamma$ on the ActivityNet Captions.**

|$\gamma$ | R@1, IoU@0.3 | R@1, IoU@0.5 | R@1, IoU@0.7  |  R@5, IoU@0.3 | R@5, IoU@0.5  | R@5, IoU@0.7 | mIoU  |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
|  5.0 | 75.76  | 60.80 | 39.23 | 91.14 | 85.43 | 74.33 | 55.51 | 
| *10.0* | *75.85*  | *60.75* | *38.88* | *91.16* | *85.34* | *74.01* | *55.47* |
| **25.0** | **75.87**  | **60.77** | **39.30** | **91.16** | **85.23** | **74.06** | **55.52** | 
| 50.0 | 75.84  | 60.98 | 38.83 | 91.04 | 85.27 | 73.98 | 55.51 |
| 75.0 | 75.74  | 60.57 | 38.63 | 90.98 | 85.26 | 73.86 | 55.29 |
| avg. | 75.81  | 60.77 | 38.97 | 91.10 | 85.31 | 74.05 | 55.46 |
| std. | 0.06   |  0.15 |  0.28 | 0.08  |  0.08 | 0.17  | 0.10 | 


> **Q3: Ablation in Fig 5 shows mostly similar results especially on NLVG, indicating that the designs in Dynamic Filter Graph actually do not quite matter.**

**A3:**
Thank you for your valuable feedback. Fig 5 in the main paper may cause your misunderstanding. Please allow us to provide additional results below (**Table R5**) to address your concerns. 
As shown in **Table R5**, our DTFNet consistently achieves the best performance compared to other graph settings, such as achieving 38.88 at the metric R@1, IoU@0.7. 

*And, why are the results of these graph methods relatively close for NLVG task not as well as the performance improvement of SLVG?*
There may be different characterizations of original textural and audio features. 
In our experiments, we use DistilBERT as textual features and Data2vec as audio features, where DistilBERT is an acknowledged high-quality textual features. 
For the NLVG task, the performance improvement with high-quality features brought by the method development is approaching saturation.
In this case, while the R@1, IoU@0.3 increases by 2.36% and the mIoU increases by 1.92% compared to GAT, they indeed mean a significant improvement for NLVG. 
As for SLVG, the audio-video interaction has the potential to be developed deeply. 

**Table R5: Different dynamic modeling methods on the ActivityNet Captions for NLVG task.**

|Method | R@1, IoU@0.3 | R@1, IoU@0.5 | R@1, IoU@0.7  |  R@5, IoU@0.3 | R@5, IoU@0.5  | R@5, IoU@0.7 | mIoU  |
| :---------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| GCN | 73.33  | 58.14 | 38.57 | 89.86 | 84.52 | 73.22 | 53.53 |
| GAT | 73.49  | 58.78 | 38.32 | 89.92 | 84.60 | 72.21 | 53.55 |
| D   | 74.40  | 59.16 | 38.46 | 90.42 | 84.62 | 73.26 | 54.06 |
| MLP | 74.29  | 59.52 | 38.07 | 90.52 | 84.48 | 72.88 | 54.12 |
| **Our DTFNet** | **75.85**  | **60.75** | **38.88** | **91.16** | **85.34** | **74.01** | **55.47** |



The Response to Reviewer Z5of (Part 3/3)

> **Q4: Template: (1) The first page is missing a header. (2) Please change  cite{..} to citep{..} for clarity.**

**A4:**
Thank you for pointing out the format issues. We have carefully proofread our paper and fixed them. 

We hope our response addresses your concerns. If there are any further questions, please do not hesitate to let us know. Thank you for your time and valuable feedback. 






