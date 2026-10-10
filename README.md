# PyTorch Neural Network for Iris Classification

A deep learning project implementing a **Multi-Layer Perceptron (MLP)** from scratch using **PyTorch** for multi-class classification on the Fisher's Iris dataset.

---

## 📌 Project Overview

This project demonstrates foundational deep learning concepts implemented with standard PyTorch idioms:
1. **Custom `nn.Module` Architecture**: Defining input, hidden, and output dense layers with non-linear activation functions.
2. **Forward Propagation**: Chaining linear transformations through Rectified Linear Unit (`ReLU`) activations.
3. **Loss Function**: Using `CrossEntropyLoss` (which combines `LogSoftmax` and `NLLLoss` for numerical stability).
4. **Gradient Optimization**: Backpropagation via `torch.optim.Adam` optimizer.
5. **Epoch-by-Epoch Training & Loss Tracking**: Monitoring convergence, loss decline, and validation accuracy.

---

## 🔬 Network Architecture

```text
Input Layer (4 features)
        │
    Linear (4 -> 7) + ReLU
        │
    Linear (7 -> 9) + ReLU
        │
    Linear (9 -> 3) [Logits]
        │
Output Layer (3 classes)
```

### Mathematical Formulation

#### 1. Layer Transformations
$$z_1 = W_1 x + b_1, \quad a_1 = \max(0, z_1)$$
$$z_2 = W_2 a_1 + b_2, \quad a_2 = \max(0, z_2)$$
$$\hat{y}_{\text{logits}} = W_3 a_2 + b_3$$

#### 2. Cross-Entropy Loss
$$\mathcal{L}(y, \hat{y}) = - \sum_{k=1}^{K} y_k \log \left( \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{K} \exp(\hat{y}_j)} \right)$$

#### 3. Weight Updates via Adam Optimizer
$$w_{t} = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 📊 Dataset

* **Source**: Classic Iris dataset (Fisher, 1936).
* **Samples**: 150 instances.
* **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width.
* **Classes**: `Setosa`, `Versicolor`, `Virginica`.

---

## 🛠️ Tech Stack & Requirements

* **Python 3.x**
* **PyTorch** (`torch`, `torch.nn`)
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-Learn**
* **Jupyter Notebook**

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RishiRaj1112/PyTorch-Iris-Neural-Network.git
   cd PyTorch-Iris-Neural-Network
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook PyTorch_Iris_NN.ipynb
   ```

---

## 📈 Key Insights

* **Tensor Operations**: Seamless data transformation from Pandas DataFrames to PyTorch `FloatTensor` and `LongTensor`.
* **Fast Convergence**: Thanks to the Adam optimizer and ReLU activations, the MLP converges to near-zero training loss within fewer than 100 epochs.

---

## 👤 Author
* **Rishi Raj Talari** - [GitHub Profile](https://github.com/RishiRaj1112)
