The Response to Reviewer 1gED (Part 1/2)

Thanks for your recognition of our proposed Dynamic Temporal Filter Network, and your insightful reply is very helpful. We follow your advice and add the new experimental results about comparing multi-query input and single-query input.

> **Q1: In ResMLP, visual features and multiple query features are concatenated and fed into the network, largely leveraging the information leakage between different queries (because the features incorporate more accurate textual information that describes the video content). If each query is individually input into the network, would this method exhibit a significant performance degradation?**

**A1:** Thank you for your valuable feedback. Here, we attempt to clarify our conclusion from the following two aspects. On one hand, regardless multi-query or single-query modes, the role of Multilayer perceptron with residual design (ResMLP) is designed to capture the associations between query and video, simulating the brain's processing of multi-source information in the early stage of visual perception for the video. On the other hand, yes, we agree with your opinion that multiple queries can indeed provide more semantics to the video and significantly fill the gap between vision-language modalities, aiding in understanding video content.  
To eliminate your concern, we have supplemented the relevant experiments and found two facts in **Table R1**:

- Our model still performs best in single-query input mode, compared to other single query methods. For example, our R@1, IoU@0.7 is 32.25, exceeding the current SOTA methods.

- At present, there are some multi-query methods, but their advantages over the single-query methods are not obvious. How to better utilize multiple queries to assist in video grounding is also a challenge. Our method has advantages in handling both single-query and multi-query input modes. 

**Table R1: Model size vs. Infer. Speed vs. Performance comparison in different query input modes.**

|(#Query) | Method | Model Size | Infer. Speed (s/Query) | R@1, IoU@0.3 | R@1, IoU@0.5 | R@1, IoU@0.7  |  R@5, IoU@0.3 | R@5, IoU@0.5  | R@5, IoU@0.7 | mIoU  |
| :---------- | :---------- | :---------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------:| 
| Single | 2D-TAN  |  21.62M  |  0.061 | 59.45 | 44.51 | 26.54 | 85.53 | 77.13 | 61.96 | - |
| Single | MS-2D-TAN  | 479.46M  |  0.141 | 61.04 | 46.16 | 29.21 | 87.30 | 78.80 | 60.85 | - |
| Single | MSAT       |  37.19M  |  0.042 |   -   | 48.02 | 31.78 |   -   | 78.02 | 63.18 | - |
| **Single** |**UNiSDNet**|**76.52M**|**0.009** |**68.66**|**52.35**|**32.25**|**89.74**|**83.35**|**70.61**|**50.22**|
| Multi  | MMN       | 152.22M | 0.014 | 65.05 | 48.59 | 29.26 | 87.25 | 79.50 | 64.76 | - |
| Multi  | PTRM      | 152.25M | 0.038 | 66.41 | 50.44 | 31.18 |   -   |   -   |   -   | 47.68 |
| **Multi**  | **UNiSDNet**  |**76.52M**|**0.009**|**75.85**|**60.75**|**38.88**|**91.16**|**85.34**|**74.01**|**55.47**|


The Response to Reviewer 1gED (Part 2/2)

> **Q2: In the ablation study, individually employing the static network and DTFNet yields significant improvements compared to the baseline. However, the combination of both modules does not exhibit a notably large improvement compared to using either single module. Is there a specific explanation for this phenomenon? The authors should provide more details about the baseline models.**

**A2:** Thank you for raising the concerns about the main network modules. 
In our manuscript, we propose a method that starts with static global review queries and video content (ResMLP), and then dynamically filters video content to extract important information (DTFNet). 
On the one hand, observing the experimental results of our static module alone (ResMLP) compared with existing works (in Tables 1 and 4 of our manuscript), previous works have overlooked and performed insufficiently in extracting information from global static reviews.
On the other hand, even if the static review information is missed and only basic cross-modal information is obtained, we achieve good video grounding results using the dynamic module (DTFNet) alone. **Effective information filtering method (DTFNet)** can still extract useful information.

For convenience, we restate Table 4 of our manuscript in the following **Table R2**. Relatively speaking, the large performance improvement of a single module through the respective exploration of multimodal information has reached **saturation** (for NLVG task, the R@1, IoU@0.3 of ResMLP and DTFNet are 73.57 and 74.56, respectively.). 
In this case, the R@1, IoU@0.3 of a single module increases from 73.57/74.56 to the final static-dynamic combination of 75.85, which is not a small improvement. 
That's why the effect of combining ResMLP and DTFNet is not as much improved as employing them alone.
This further demonstrates the effectiveness of our static and dynamic modules. 

**Table R2: Ablation studies on the static and dynamic modules on the ActivityNet Captions and ActivityNet Speech datasets for both NLVG and SLVG.**
|Task | Static Module | Dynamic Module | R@1, IoU@0.3 | R@1, IoU@0.5 | R@1, IoU@0.7  |  R@5, IoU@0.3 | R@5, IoU@0.5  | R@5, IoU@0.7 | mIoU  |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| NVLG | N | N | 61.22 | 44.46 | 26.76 | 87.19 | 78.63 | 63.60 | 43.98 |
| NVLG | Y | N | 73.57 | 58.70 | 37.07 | 91.17 | 85.55 | 73.98 | 54.06 |
| NVLG | N | Y | 74.56 | 59.45 | 37.44 | 90.98 | 85.43 | 73.60 | 54.43 |
| **NVLG** | **Y** | **Y** | **75.85** | **60.75** | **38.88** | **91.16** | **85.34** | **74.01** | **55.47** |
| SLVG | N | N | 53.63 | 35.91 | 20.51 | 84.71 | 74.21 | 55.95 | 38.23 |
| SLVG | Y | N | 69.71 | 53.75 | 31.26 | 90.42 | 84.11 | 70.82 | 50.69 |
| SLVG | N | Y | 71.34 | 54.03 | 31.51 | 89.75 | 82.62 | 68.12 | 50.97 |
|**SLVG** | **Y** | **Y** | **72.27** | **56.29** | **33.29** | **90.41** | **84.28** | **72.42** | **52.22** |


**About the baseline model**, we belong to the scope of the 2D proposal-based method. Specifically, we adopt the same backend decoding method as MMN (including the 2D proposal generation module). For training losses, we use the cross entropy loss between the predicted 2D proposal score map and groundtruth which is consistent with 2D-TAN, and add the contrastive loss proposed in MMN as an auxiliary constraint. That is to say, our method is based on the typical 2D temporal map's backend decoding architecture (2D-TAN and MMN), and by incorporating a static-dynamic module design as the fore-end implementation. 
Our work further develops the 2D proposal-based method in this field. 

We hope our response addresses your concerns. If there are any further questions, please let us know. Thank you for your time and valuable feedback. 

