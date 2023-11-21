
The Response to Reviewer TT77 (Part 1/3)

Thanks for your insightful and constructive feedback on our manuscript. Your positive reception of our visual perception biological motivation to guide network design and recognition of our method and experiments are both encouraging and deeply appreciated. We will make some clarifications in accordance with your suggestions.

  
> **Q1: The introduction reads like a related work. It will be great to make more comparison between this work and previous work. Answering what is wrong with previous works? and where the efficiency and performance gain come from in this paper?**

**A1:** Thanks for your valuable suggestion. Technology-wise, previous methods mostly focus on solving a certain aspect of the Temporal Video Grounding (TVG) task, such as representation learning of language and video self-modality [R1, R2], multimodal fusion [R3, R4], cross-modal interaction [R5, R6], proposal candidate generation [R7, R8], proposal-based cross-modal matching [R9, R10], target moment boundary regression [R11, R12], and so on.
Our work actually proposes a new paradigm to establish a two-stage unified static and dynamic semantic complementary new architecture. Its unique characteristics are that
- Processing multimodal signals in a unified ResMLP network, while many previous works are independently encoding the language modality and video modality [R1-R7, R9-R12].
- After the implementation of the above static ResMLP, we introduce a Gaussian nonlinear filtering method to learn the semantic associations within the video and combine it with the back-end proposal generation to promote cross-modal semantic alignment, further developing the proposal-based TVG method.

Our work is inspired by visual perception biology. This unified static and dynamic two-layer architecture performs excellent joint learning of language and video, which achieves state-of-the-art performance on NLVG and SLVG tasks. 
The ablation experiments in **Table 4** of our manuscript demonstrate the effectiveness of each proposed component of this work. We will improve our manuscript based on your constructive suggestions.

**Eﬀiciency and performance:** Technology-wise, the inference efficiency of our model comes from a streamlined architecture design, which allows for parallel inference of video and multiple queries in a unified ResMLP, saving time and expenses.
Our promising model performance mainly comes from our model's comprehensive information interaction with query and video (ResMLP, static network), as well as our detailed inference of video content (DTFNet, dynamic graph filtering network), which was confirmed in the first ablation experiment in **Table 4** of the manuscript.

**References:**

[R1] Xia et al., Video-guided curriculum learning for spoken video grounding. ACM MM, pp. 5191–5200, 2022.

[R2] Rodriguez et al., Memory-efficient temporal moment localization in long videos. ECAL, pp. 1901–1916, 2023.

[R3] Li et al., Proposal-free video grounding with contextual pyramid network. AAAI, pp, 1902–1910, 2021. 

[R4] Liu et al., Exploring optical-flow-guided motion and detection-based appearance for temporal sentence grounding. IEEE TMM, 2023.

[R5] Liu et al., Jointly cross-and self-modal graph attention network for query-based moment localization. ACM MM, pp. 4070–4078, 2020.

[R6] Sun et al., Video moment retrieval via comprehensive relation-aware network. IEEE TCSVT, 2023.

[R7] Zhang et al., Learning 2d temporal adjacent networks for moment localization with natural language. AAAI, pp. 12870–12877, 2020.

[R8] Zhang et al., Multi-stage aggregated transformer network for temporal language localization in videos. CVPR, pp. 12669–12678, 2021. 

[R9] Gao et al., Fast video moment retrieval. ICCV, pp. 1523–1532, 2021.

[R10] Zheng et al., Phrase-level temporal relationship mining for temporal sentence localization. AAAI, pp. 3669–3677, 2023.

[R11] Zhang et al., Natural language video localization: A revisit in span-based question answering framework. IEEE TPAMI, 44(8), pp. 4252–4266, 2021.

[R12] Liu et al., Skimming, locating, then perusing: A human-like framework for natural language video localization. ACM MM, pp. 4536–4545, 2022.


  
The Response to Reviewer TT77 (Part 2/3)

> **Q2: This paper introduces some new/confusing terminologies with their own definition, which hurts the reading experience. For example, 'static semantic supplement network' and 'activity-silent mechanism' are actually the global context interaction.**

**A2:** Thanks for your friendly reminder. To clear up your confusion, we clarify and revise some confusing representations as below.

**1) 'static semantic supplement network':** This is the network naming in terms of function, because in our work, we focus on understanding video content in a multimodal environment [R13], and we adopt a unified framework of static and dynamic structures. In the early stage, this global interaction mode first perceives all multimodal information, and then information filtering is performed. In terms of the functionality of the static network for video understanding, it provides more video descriptions information and significantly fills the gap between vision-language modalities, aiding in understanding video content.

**2) 'activity-silent mechanism':** This is the mechanism by which the brain processes information in the early stages of human visual perception for the video, as mentioned in [R14]. It manifests as a static multi-source information interaction, achieving the 'global broadcast communication' of the brain. Because we are strongly inspired by this mechanism in static network design, we specifically mentioned this professional biological term in our manuscript. This network can achieve the effect of 'the global context interaction' from a technical perspective, and this is the basic intention for our network design. 

We will incorporate your suggestion and consider both motivation and technical introduction in the paper.

**References:**

[R13] Lisa et al., Localizing moments in video with natural language. CVPR, pp. 5803-5812, 2017.

[R14] Barbosa et al., Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory. Nature neuroscience, 23(8), 1016-1024, 2020. 


> **Q3: Although the motivation of static and dynamic network is demonstrated, the justification of specific design is not enough. For example, in the static network, transformer architecture or the recent S4[1] architecture can also be used as long-range filter. Some ablation studies regarding either the performance or efficiency would be great to include.
[1] Efficiently modeling long sequences with structured state spaces. ICLR 2021.**


**A3:**
Thank you for your valuable suggestion.
For static perception modeling of multi-source information, based on the theoretical guidance of biology, we consider using Multilayer perceptron (MLP) to handle language and video uniformly.
The reviewer suggested we consider more long-range filters in our studies, therefore, we have added the ablation experiments about our static network, and the results are shown in **Table R3**:

**Table R3: Comparison of different static networks on ActivityNet Captions for NLVG task.**

| Static | Infer. Speed (s/query) | R@1,IoU@0.3 | R@1,IoU@0.5 | R@1,IoU@0.7  |  R@5,IoU@0.3 | R@5,IoU@0.5  | R@5,IoU@0.7 | mIoU  | 
| :---------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Transformer | 0.024 | 75.17 | 59.98 | 38.38 | 91.26 | 85.77 | 74.25 | 54.97 |
| S4          | 0.030 | 70.41 | 55.11 | 34.93 | 89.12 | 83.16 | 70.54 | 51.40 |
| **Our ResMLP** | **0.009** | **75.85** | **60.75** | **38.88** | **91.16** | **85.34** | **74.01** | **55.47**| 

The results indicate that: 
- ResMLP achieves model performance/efficiency trade-offs.
- Note that the S4 model is particularly skilled in handling very long (about 1w) sequence tasks, but in the ActivityNet Captions dataset of NLVG tasks, the total number of video clips and queries does not exceed 100, which may be the reason why it is not suitable for our task.
- The performance of Transformer is close to ResMLP, but its 'Infer. Speed' cost is 2.67 times that of ResMLP. In terms of better performance and efficiency, we choose ResMLP as the implementation of the static network.

  
> **Q4: No notation for the 'FNN'. Is this the feedforward network?**

**A4:** Yes, it is. 'FNN' is the abbreviation of feedforward network. We have unified the expression of FNN and added a full name explanation in the paper.


> **Q5: In the Figure 5, no notation/description for 'D'.**

**A5:**
The following is a description of **D**, and we will provide a clearer explanation of **D** in the paper.
In the setting of **D**, we use the message aggregation wight $f_{ij}=1/(d_{ij}+1)$ to replace $f_{ij}$=$\mathcal{F}\_{filter}(d_{ij})$, which indicates that we still consider the clue of temporal distance $d_{ij}$ between two nodes but remove the entire Gaussian filtering calculation from our method. 
This replacement results in a decrease of 0.42 and 1.92 on R@1, IoU@0.7 for NLVG and SLVG, respectively. 

  
The Response to Reviewer TT77 (Part 3/3)

> **Q6: In the dynamic network, not sure why use Gaussian filter on the distance ($d_{ij}$). Can you provide more insights? why not directly use the distance.**


**A6:**
Thank you for your valuable feedback. In our consideration, using a Gaussian filter is necessary for the following reasons. 

- In the process of human visual perception of the video, the processing of temporal relationships between video clips is complex, we must choose a non-linear modeling approach.
-  Brain processing of temporal relationships between video clips has four characteristics: 1) Nonlinearity; 2) High-dimensional; 3) Short-term Effect; 4) Relative Temporal Distance Decay. The current video clip has $h$ Gaussian filters on both temporal directions that gradually decrease in correlation, which can reflect the visual persistence phenomenon when people browse videos. That is, the recent past has a significant impact on their perception of the present.

Due to the **dynamic nature, continuity (high-dimensional), and nonlinearity (complexity)** of visual perception transmission, the discrete scalar $d_{ij}$ is insufficient to express these characteristics. 
Therefore, we use filter generating networks to generate filters dynamically. 
The Gaussian function has already been exploited in deep neural networks, such as Gaussian kernel grouping and Gaussian radial basis function. These Gaussian functions have been proven to be effective in simulating the high-dimensional nonlinear information in various scenes. 
Inspired by these works, we use multi-kernel Gaussian radial basis to extend the influence of $d_{ij}$ into high-dimensional space, thereby reflecting the continuous complexity of the perception process. Meanwhile, using a multi-kernel Gaussian with different biases can avoid a plateau at the beginning of training due to the highly correlated Gaussian filters. 


  
> **Q7: Is there any chance also leverage the audio signal into this work, formulating a multi-model graph?**

**A7:** 
Thank you for your valuable suggestion. Your suggestion is very insightful and prescient. Yes, we agree that there is a chance. The original intention of our work is to solve the query-driven video content understanding, so we choose to model the video content in the graph after the global view with the ResMLP module, and the experiments achieve well results.
In further research, if we interact with the original audio signal and video, we may have to perform multi-task learning with mutual guidance between modalities, such as video-guided audio graph learning, audio-guided video graph learning, and audio-video joint graph learning to promote video event understanding (localization). More utilization of semantic consistency and complementarity between modalities is needed. 
We will continue to study the possibility of the audio-video multi-model graph in our future work, which will be very meaningful. 

We hope our response addresses your concerns. If there are any further questions, please do not hesitate to let us know. Thank you for your time and valuable feedback.

