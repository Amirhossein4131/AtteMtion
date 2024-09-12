# 🌟 **In-Context Learning of Physical Properties: Few-Shot Adaptation to Out-of-Distribution Molecular Graphs** 🌟

[![arXiv](https://img.shields.io/badge/arXiv-2406.01808-B31B1B.svg)](https://arxiv.org/abs/2406.01808)

> *In-Context Learning of Physical Properties: Few-Shot Adaptation to Out-of-Distribution Molecular Graphs*

---

## 📜 **Abstract**
> *Large language models manifest the ability of few-shot adaptation to a sequence of provided examples. This behavior, known as in-context learning, allows for performing nontrivial machine learning tasks during inference only. In this work, we address the question: can we leverage in-context learning to predict out-of-distribution materials properties? However, this would not be possible for structure property prediction tasks unless an effective method is found to pass atomic-level geometric features to the transformer model. To address this problem, we employ a compound model in which GPT-2 acts on the output of geometry-aware graph neural networks to adapt in-context information. To demonstrate our model's capabilities, we partition the QM9 dataset into sequences of molecules that share a common substructure and use them for in-context learning. This approach significantly improves the performance of the model on out-of-distribution examples, surpassing the one of general graph neural network models. *

---



## 📚 **Citation**

If you use this work in your research, please cite:

```bibtex
@misc{kaszuba2024incontextlearningphysicalproperties,
      title={In-Context Learning of Physical Properties: Few-Shot Adaptation to Out-of-Distribution Molecular Graphs}, 
      author={Grzegorz Kaszuba and Amirhossein D. Naghdi and Dario Massa and Stefanos Papanikolaou and Andrzej Jaszkiewicz and Piotr Sankowski},
      year={2024},
      eprint={2406.01808},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.01808}, 
}

