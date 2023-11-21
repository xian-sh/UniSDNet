

The Response to Reviewer GE2s (Part 1/3)

We are so encouraged and cheerful to receive your constructive comments, your positive reception of UniSDNet's biological motivation and recognition of our two-stage information aggregation methods are both encouraging and deeply appreciated. 

> **Q1: What is the motivation behind the implementation of Static Semantic Supplement Network? I am wondering how the cross-modal interaction is performed through the MLP layers. To my understanding, the shared weights across different modalities would extract some common features spanning different modalities. Some analytical experiments on this would be beneficial. Also, the architecture design seems similar to that of Transformer blocks except for the self-attention. What happens if we use the conventional Transformer layers?**

**A1:**
Thank you for raising the concerns about the motivation of static network. We would like to explain the implementation of our static network from the perspectives of motivation, technology, and experiment.

- **Motivation-wise,** the implementation of Static Semantic Supplement Network is strongly inspired by the **global neural workspace (GNW)** theory in biology [R15], that is, for video understanding, the brain will perform ``global broadcast communicate'' (interaction) of the multi-source information in the early stage of visual perception for the video, while the cognitive consciousness of the brain will \textbf{strengthen perceptual conditions} and \textbf{delay conditioned reflexes}, enabling the human brain to understand new conditioned reflexes brought about by perceptual information updates.

On the premise of the above theory, this inspires us to utilize residual networks structure to update our understanding of video sequences in the delaying conditioned reflex states by aggregating the perceptual information. \textbf{Multilayer perceptron (MLP)} is proven to be effective and has strong interpretability in modeling multi-level cognitive processing [R19], thus we use MLP to aggregate the perceptual information. The visual consciousness process is modeled as follows：
$x^{n+1} = \text{ResMLP}(x^n) = x^n+\text{MLP}(x^n),$

where $x^n$ represents the perception condition in the current state, $x^{n+1}$ denotes the reflection result,  % under the perception condition, and $\text{ResMLP}(x^n)$ can enhance the information of the perception condition $x^n$.

- **Technology-wise**,  we concatenate multiple queries and video clips together as a sequence input into the network for fully connected interaction of all tokens, its main function is to supplement and associate semantics between modalities before the latter graph filtering, provide more video descriptions information and significantly fill the gap between vision-language modalities, aiding in understanding video content.

- **Experiment-wise**, as per your valuable suggestion, we have tested the effect of Transformer as a static network as shown in the table below (**Table R6**), and as you are concerned, Reviewer ``TT77'' has also focused on the ablation study of the static module. From the results, in terms of performance and efficiency, Transformer is close to our method, but our results are better. We speculate that the reason is that our network also includes the second stage of graph filtering. The static network uses a lightweight and stable network, which is more conducive to model training. Using Transformer as a static network increases the weight and instability factors of the network. 

**Table R6: Results of different static networks on the ActivityNet Captions.**

|Static | Infer. Speed (s/query) | R@1, IoU@0.3 | R@1,IoU@0.5 | R@1,IoU@0.7  |  R@5,IoU@0.3 | R@5,IoU@0.5  | R@5,IoU@0.7 | mIoU  | 
| :---------- | :----------: | :----------:  | :----------: | :----------:  | :----------:  | :----------:  | :----------:  | :----------: | 
| Transformer | 0.024 | 75.17 | 59.98 | 38.38 | 91.26 | 85.77 | 74.25 | 54.97| 
|**Our ResMLP** | **0.009** | **75.85**	| **60.75**	| **38.88**	| **91.16** | **85.34** | **74.01**	| **55.47** |

**References:**

[R15] Volzhenin et al., Multilevel development of cognitive abilities in an artificial neural network. Proceedings of the National Academy of Sciences, 119(39), e2201304119, 2022.

[R19] Chavlis et al., Drawing inspiration from biological dendrites to empower artificial neural networks. Current opinion in neurobiology. 70, 1-10, 2021. 


The Response to Reviewer GE2s (Part 2/3)

> **Q2: The proposed architecture exploits multiple queries at once, to facilitate the model learning. However, how the number of queries affects the performance is not diagnosed. An ablative study on the number of queries regarding performance and cost would be helpful.**

**A2:**
Thanks for your kind and valuable suggestion. Following your suggestions, we have conducted ablation studies on the impact of query quantity on model performance on the ActivityNet Captions dataset. An explanation of the distribution of query quantity per video is that, the training set of this dataset contains 10,009 videos, with a minimum of 2 and a maximum of 27 queries per video. Due to the number of queries for each video in this dataset being concentrated in [3, 8], we set the upper limits for the number of queries $M$ per video feed as 1, 3, 5, 8, and no upper limit. In fact, the 25\%, 50\%, and 75\% quantiles are 3, 3, and 4, respectively. When the $M$ = 8, 97.33\% of samples in the dataset are already included. 

The results are shown in **Table R7** below, and according to the experiments, when the number of training query inputs $M=1$, our model is comparable with state-of-arts. As the query number upper limit increases, the performance of our model improves, which demonstrates the effectiveness of our model in utilizing multimodal information.

**Table R7: The impact of query quantity on ActivityNet Captions dataset for NLVG task.**

|Method (\#Query) | R@1,IoU@0.3 | R@1,IoU@0.5 | R@1,IoU@0.7  |  R@5,IoU@0.3 | R@5,IoU@0.5  | R@5,IoU@0.7 | mIoU  | 
| :---------- | :----------:  | :----------:  | :----------:  | :----------:  | :----------:  | :----------:  | :----------: |
| UniSDNet ($M=1$)     | 68.66 | 52.35 | 32.25 | 89.74 | 83.35 | 70.61 | 50.22 |
| UniSDNet ($M\leq 3$) | 72.42 | 57.30 | 36.64 | 89.82 | 83.88 | 72.14 | 53.30 |
| UniSDNet ($M\leq 5$) | 74.41 | 59.27 | 37.70 | 90.83 | 85.07 | 74.17 | 54.58 |
| UniSDNet ($M\leq 8$) | 74.79 | 60.28 | 38.35 | 90.68 | 85.15 | 73.90 | 55.03 |
| UniSDNet (Full)      |**75.85**| **60.75** | **38.88** | **91.16** | **85.34** | **74.01** | **55.47** | 

> **Q3: In Figure 5, the effectiveness of the proposed filtering GCN is clearly verified. On the other hand, there are some interesting tendency differences between NLVG and SLVG. That is, the graph convolution layer itself is important, yet different layer modeling brings insignificant performance gaps on NLVG. In contrast, on SLVG, the graph modeling brings negligible gains alone, but the proposed filtering mechanism shows substantial improvements. How can one interpret this phenomenon? If you have, please share some insights.**

**A3:** 
Thank you for your detailed comments. Compared to other graph methods, our dynamic model exhibits interesting inconsistencies in NLVG and SLVG tasks. We hope to provide our own humble opinion on this phenomenon.
We infer that a large part of this is due to the different feature characteristics of text and audio.
In our experiments, we use DistilBERT as textual features and Data2vec as audio features, where DistilBERT is an acknowledged high-quality textual features. 
For the NLVG task, the performance improvement with high-quality features brought by the method development is approaching saturation.
In this case, while the "R@1, IoU@0.7" indicator increases by 1\%, it indeed means a significant improvement for NLVG. 
As for SLVG, current works are relatively rare, and the audio-video interaction has the potential to be developed deeply. 


The Response to Reviewer GE2s (Part 3/3)

> **Q4: The proposed method is well validated in the datasets with one-to-one matching between queries and moments. How would it perform for one-to-many matching datasets, such as QVHighlights [R20]?**

**A4:**
Thanks for pointing out the excellent work of QVHighlights and constructive comments. 
QVHighlights is a recently publicized dataset for both moment retrieval (MR) and highlight detection (HD) tasks.  
Following the practice [R20, R21], %we use train split for training and val split for testing. 
the commonly used metric is Recall@K, IoU=[0.5, 0.7].
As your valuable suggestion, we have tested our model on this %one-to-many matching 
dataset. Current works on the QVHighlights dataset have undergone multi-task learning, including both MR and HD tasks. 
However, due to the urgent rebuttal deadline, we only complete our model evaluation for the MR single task mode on this dataset. 
We have listed the reported results of exiting works for the MR single task in **Table R3** and compare ours with them. From the results and facts, there are some conclusions. 

- Our method still performs the best with signle MR task test in one-to-many query matching mode.
This also proves that the good universality of our model for the task setup of one query to multiple moments retrieval, in which each query corresponds to multiple moments retrieval.

- To be applicable to both HD and MR tasks, M-DETR [R1] and UMT [R2] are extended on the basis of the DETR method. We adopt the typical 2D temporal proposal infrastructure and our methods still performs well in solving one-to-many matching MR task. This demonstrates our model's advantage of high precision (R@1, IoU@0.7 is 40.39, much higher than the 33.82 of UMT method).

- In the future, we will continue to collaborate with HD tasks, whose multi-task joint training method has been proven to further promote the MR results [R21], and finish this comparison with fair conditions and believe that we will have better performance.



**Table R8: Comparison with the state-of-art for one-to-many query matching on QVHighlights for single MR task.**
| Method       |        Venue | Task | Video | R@1,IoU@0.5 | R@1,IoU@0.7  | 
| :---------- | :----------  | :----------:  | :----------:  | :----------:  | :----------:  | 
| M-DETR [R20] | NeurIPS 2021 | Single MR task | Slowfast + CLIP | 44.84   |    25.87    | 
| UMT [R21]    | CVPR 2022    | Single MR task | Slowfast + CLIP | 54.14   |    33.82    | 
| Ours         |              | Single MR task | Slowfast + CLIP | 54.58   |    40.39    | 

**References:**

[R20] Lei et al., QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries. NeurIPS, pp. 11846--11858, 2021.

[R21] Liu et al., Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. CVPR, pp. 3042-3051, 2022.


> **Q5: The manuscript contains some formatting errors due to the excessively small margins between captions and the main text.**

**A5:** Thank you for pointing out the format issues. We have carefully proofread our paper and fixed them. 

We hope our response addresses your concerns. If there are any further questions, please do not hesitate to let us know. Thank you for your time and valuable feedback.
