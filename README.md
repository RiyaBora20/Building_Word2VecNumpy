
---
## 1. Approach: Skip-gram with Negative Sampling (SGNS)

My implementation uses the **Skip-gram** architecture. Unlike the Continuous Bag of Words (CBOW) which predicts a target word from its context, Skip-gram uses a single **center word** to predict its surrounding **context words**.

---
## 2. Dataset

I used **"Alice in Wonderland"** by Lewis Carroll (via Project Gutenberg).

* **Preprocessing:** The text was lowercased and stripped of non-alphabetic characters.
* **Filtering:** I implemented a `min_count` threshold (5), removing rare words to focus on meaningful semantic patterns.
* **Size:** The final corpus consists of roughly 26,000 tokens with a filtered vocabulary of ~684 unique words.


---

## 3. Model Architecture

The model is a **shallow neural network** consisting of an input layer, a single hidden layer (the embedding), and an output layer.
### The Components

* **Input:** A single **Center Word**. In the code, this is passed as an integer ID.
* **Hidden Layer (Embedding):** This layer acts as a "lookup table." It transforms the word ID into a dense, numerical vector of size `embed_size` (e.g., 300). This vector represents the word's position in our high-dimensional semantic space.
* **Output Layer:** This layer compares the center word vector against the **Context Word** vector. It produces a single probability score using a Sigmoid activation.

### The Data Flow

1. **Selection:** Pick a center word and a potential context word.
2. **Projection:** The center word is "projected" into the embedding space by retrieving its row from the `W_in` matrix.
3. **Similarity Check:** Calculate the **dot product** between the center word vector and the context word vector from the `W_out` matrix.
4. **Prediction:** The result is passed through a **Sigmoid** function to output a probability between **0 and 1**.
---
## 4. Optimization Procedure

The training loop is a manual implementation of Stochastic Gradient Descent (SGD).

### Forward Pass

For every center word ($v_c$) and context word ($u_o$), we compute the dot product:


$$z = v_c \cdot u_o$$


This raw score is passed through the **Sigmoid function** to produce a probability:


$$P(context|center) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Loss Function

We minimize the **Binary Cross Entropy (BCE)** loss. For a single positive pair and its negative samples, the loss is:


$$J = -\log(\sigma(v_c^T u_{pos})) - \sum_{i=1}^{k} \log(\sigma(-v_c^T u_{neg_i}))$$

### Gradient Calculation

The gradient for the dot product $z$ simplifies to the **prediction error**:


$$\frac{\partial J}{\partial z} = (\text{prediction} - \text{truth label})$$


Using the chain rule, we derive the gradients for our vectors:

* $\frac{\partial J}{\partial u_o} = (\sigma(z) - y)v_c$
* $\frac{\partial J}{\partial v_c} = (\sigma(z) - y)u_o$

### Parameter Updates

We update the weight matrices `W_in` and `W_out` by moving them in the opposite direction of the gradient, scaled by the learning rate ($\eta=0.025$):


$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_{\theta} J$$

## 4. Results

* **Convergence:** The loss consistently decreased across 10 epochs (e.g., from ~3.0 to ~1.4 for `embed_size=300`).
* **Semantic Relationships:** Using vectorized Cosine Similarity, the model successfully clustered related tokens. For example, `good` showed high similarity to other words like `reason (Score: 0.2738)` & `great (Score: 0.2685)`.
* **Dimensionality Impact:** Increasing `embed_size` resulted in lower final loss, demonstrating the increased capacity of the model to capture complex relationships.

---
