# How do we make audio caption datasets?

In the [VGCL paper](https://arxiv.org/pdf/2209.00277.pdf), 58 volunteers are asked to fluently read the text in a clean surrounding environment to obtain the sudio dataset corresponding to AncitivyNet Caption Dataset.

However, in our paper, we use machine simulation of human voice to synthesize audio subtitle datasets corresponding to Charades-STA Caption dataset and TACoS Caption Dataset.

There are several reasons for machine simulation:

  * Thanks to the development of [TTS technology](https://huggingface.co/microsoft/speecht5_tts), it can highly simulate human voice, including other complex features such as style.
  * Cost savings.
  * Diverse vocal styles, for example, the [Matthijs/cmu-arctic-xvectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors) dataset contains around 8000 vocal features to choose from. [details](http://www.festvox.org/cmu_arctic/)

