# DA6401 - Introduction to Deep Learning
# Name : Prudhvi VVR Ajay Sreenivas
# Roll no : DA25M023
## Assignment 3: Implementing a Transformer for Machine Translation

This repository contains a full PyTorch implementation, comprehensive ablation studies, and empirical evaluations of a Sequence-to-Sequence Transformer architecture engineered for English-to-German Machine Translation. All core parameters, gradient dynamics, structural modifications, and evaluation metrics are instrumented and logged using Weights & Biases (W&B).

---

### 📊 Public Weights & Biases Report
The complete tracking dashboard, interactive attention map panels, and empirical comparative charts can be viewed through the official public report link below:

👉 **[View the Live W&B Assignment 03 Report](https://wandb.ai/da25m023-iit-madras-foundation/DA6401-A3-Transformer-v3/reports/Assignment-03-Report--VmlldzoxNjkzNDY2NQ?accessToken=625kzxt8r1oue7uwn17srl0fpte7asenrsle605x629qlp5zplzua5fseu5tuclt)**

---

## 🚀 Key Experimental Findings & Ablations

### Task 2.1: The Necessity of the Noam Scheduler
* **Objective:** Contrast a dynamic schedule utilizing a linear warmup followed by inverse square-root decay (Noam Scheduler) against a static learning rate baseline ($\text{LR} = 10^{-4}$).
* **Empirical Observations:** The fixed learning rate exhibits a sharp drop in training loss during the earliest steps due to its aggressive initial step size. However, it quickly plateaus and stagnates. The Noam Scheduler takes longer to minimize early loss because of its conservative linear warmup phase, but it ultimately establishes a highly stable trajectory, smoothly outperforming the fixed baseline in late-stage training and capturing a superior, stable validation categorical accuracy.
* **Theoretical Grounding:** Transformers have a weak inductive bias and rely completely on self-attention weights. Initializing training with static, uncalibrated high steps causes erratic gradients that risk trapping the network in severe, sub-optimal local minima. The linear warmup phase artificially subdues optimization steps to give the multi-head self-attention mechanisms structural time to stabilize before decaying safely down toward global optimization minima.

### Task 2.2: Ablation of the Scaling Factor ($1/\sqrt{d_k}$)
* **Objective:** Track and analyze the direct impact of removing the scaling denominator ($\sqrt{d_k}$) from the Scaled Dot-Product Attention component over a strict window of 1,000 steps.
* **Empirical Observations:** In the unscaled experiment, the gradient norms of the Query and Key weights (`Q_grad_norm` and `K_grad_norm`) collapse to near-zero almost instantly within the first 100 steps. Conversely, the model configured with the scaling factor maintains healthy, informative gradient norms floating steadily between $0.05$ and $0.15$ across the entire 1,000-step runtime.
* **Theoretical Grounding:** As noted in Section 3.2.1 of the *Attention Is All You Need* paper, when the key dimensionality ($d_k$) is high, the dot product values grow massively in magnitude. Passing these extreme values into a standard softmax function pushes the system into highly saturated, flat regions where the local mathematical derivative approaches zero. Incorporating the scaling factor normalizes the variance of the logits back to 1, completely preventing softmax saturation and sustaining healthy backpropagation lines.

### Task 2.3: Attention Rollout & Head Specialization
* **Objective:** Extract and inspect attention matrices derived from the final encoder layer for specific multi-head behavior using the sample sentence: *"Ein Hund rennt über das Gras."*
* **Empirical Observations:**
  * **Head Specialization:** Clear functional specialization is visible across the module. For instance, Heads 2 and 3 display distinct, sharp vertical bands centered entirely on specific structural landmarks and nouns like `hund` and `<eos>`, implying that every position in the sequence relies heavily on these specific semantic anchors. Meanwhile, Head 5 targets structural block components, focusing its distribution directly on the prepositional phrase (`über`, `das`, `gras`).
  * **Head Redundancy:** Strong evidence of mathematical redundancy exists across the heads. Specifically, **Head 6** and **Head 8** display virtually identical vertical attention concentrations, proving that independent heads within a multi-head block can converge on tracking the same properties.

### Task 2.4: Positional Encoding vs. Learned Embeddings
* **Objective:** Run a 10-epoch evaluation comparing fixed mathematical Sinusoidal Positional Encodings against a parametric, learned position lookup table (`torch.nn.Embedding`).
* **Empirical Observations:** Both configurations yield highly competitive validation performance, converging tightly around a validation BLEU score of approximately $11.5$ to $11.8$. In our targeted translation sequence context, both representations successfully map relative sequences.
* **Theoretical Challenge (Extrapolation):** Despite identical empirical scores on standard lengths, learned positional embeddings possess a structural threshold limit: if the inference module encounters a sequence longer than the maximum index specified during training, it encounters completely untrained parameters, resulting in an immediate breakdown. Sinusoidal functions operate via periodic continuous trigonometric equations, meaning they can deterministically compute unique relative distance combinations for sequences of arbitrary lengths—even those completely unseen during training.

### Task 2.5: Decoder Sensitivity & Label Smoothing
* **Objective:** Evaluate validation regularization and softmax behaviors when applying Label Smoothing ($\epsilon_{ls} = 0.1$) versus standard Cross-Entropy ($\epsilon_{ls} = 0.0$).
* **Empirical Observations:** Tracking the `Prediction_Confidence` panel reveals that the unsmoothed model drives its target token probability to an overconfident value of $0.98+$. The smoothed model restricts this peak probability, stabilizing safely within an explicit confidence window of $0.85$ to $0.88$.
* **Theoretical Grounding:** Standard cross-entropy forces the network to map outputs to strict, hard one-hot targets ($1.0$ for the correct class, $0.0$ for others). To reach a probability of $1.0$, the network must mathematically drive the correct logit value towards infinity relative to all incorrect choices, leading to radical overfitting. Label smoothing smoothly shifts a fraction ($\epsilon_{ls}$) of the probability mass uniformly to alternative options, regularizing logit inflation and significantly improving validation generalization despite a superficial increase in training perplexity.

---

## 📂 Project Directory Structure

```text
├── src/
│   ├── model.py            # Encoder-Decoder Transformer and Attention Blocks
│   ├── loss.py             # Customized LabelSmoothingLoss class
│   ├── scheduler.py        # Noam Learning Rate Scheduler framework
│   └── utils.py            # Multi-lingual Tokenization and BLEU scoring metrics
├── main.py                 # Core orchestration file managing the multi-step ablation suite
├── requirements.txt        # Python package prerequisites
└── README.md               # Main repository documentation
