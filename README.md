# GNN Over-Smoothing Analysis

This repository contains my **3rd semester academic Graph Neural Network (GNN) project**, focused on analyzing the **over-smoothing problem in deep Graph Convolutional Networks (GCNs)**.

## Project Overview

In deep GNNs, node representations can become increasingly similar as the number of layers grows. This is known as **over-smoothing**, and it can reduce the model’s ability to distinguish between nodes from different classes.

In this project, I studied how over-smoothing affects node classification performance and compared different techniques that try to reduce it.

## Objective

The main goal of this project is to analyze how increasing GCN depth impacts:

- classification accuracy
- feature similarity across layers
- feature variance across layers

I also compared multiple GCN variants to see which methods help reduce over-smoothing more effectively.

## Models Compared

The project includes experiments with the following variants:

- **Plain GCN**
- **Residual GCN**
- **PairNorm GCN**
- **DropEdge GCN**

These models were tested at different depths:

- 2 layers
- 4 layers
- 8 layers

## Dataset

The experiments were performed on the **Cora citation network dataset** using **PyTorch Geometric**.

## Metrics Used

To study over-smoothing, I used both performance-based and representation-based metrics:

- **Test Accuracy**
- **Layer-wise Feature Similarity**
- **Layer-wise Feature Variance**

### Why these metrics?

- **Higher feature similarity** across nodes can indicate that embeddings are becoming too alike.
- **Lower feature variance** can indicate that node representations are collapsing.
- Together, these help explain whether deeper GNNs are losing discriminative power.

## Visualizations Included

The notebook includes plots for:

- Test Accuracy vs Depth across variants
- Layer-wise Feature Similarity curves
- Layer-wise Feature Variance curves
- Similarity and Variance comparison plots
- Similarity heatmaps
- Variance heatmaps
- Final-layer Similarity vs Depth
- Final-layer Variance vs Depth
- Average Test Accuracy by Variant

## Tools and Libraries

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- Seaborn

## Project Structure

```bash
GNN_pjt.ipynb
README.md
```

## Key Results

Some of the best test accuracies observed in the notebook are:

| Variant   | Depth | Best Test Accuracy |
|-----------|------:|-------------------:|
| Plain     | 2     | 0.813 |
| Plain     | 4     | 0.795 |
| Plain     | 8     | 0.776 |
| Residual  | 2     | 0.817 |
| Residual  | 4     | 0.795 |
| Residual  | 8     | 0.791 |
| PairNorm  | 2     | 0.741 |
| PairNorm  | 4     | 0.676 |
| PairNorm  | 8     | 0.734 |
| DropEdge  | 2     | 0.815 |
| DropEdge  | 4     | 0.800 |
| DropEdge  | 8     | 0.760 |

### General observations

- Shallow GCNs performed well on the Cora dataset.
- As depth increased, some models showed reduced performance, indicating over-smoothing effects.
- Residual connections and DropEdge helped maintain stronger performance at deeper layers compared to the plain GCN in some settings.
- PairNorm showed weaker performance in this implementation, which makes it an interesting point for further investigation.

## What I Learned

Through this project, I learned:

- how deep GCNs behave as network depth increases
- how over-smoothing affects node embeddings
- how to evaluate GNNs beyond just accuracy
- how to compare architectural techniques such as Residual connections, PairNorm, and DropEdge
- how to use visualization to interpret internal model behavior

## How to Run

1. Clone this repository
2. Install the required libraries
3. Open the notebook and run all cells

### Install dependencies

```bash
pip install torch torch-geometric numpy matplotlib seaborn
```

## Future Improvements

This is currently an academic semester project, and I plan to improve it further. Possible next steps include:

- testing on additional datasets such as CiteSeer and PubMed
- running multiple seeds for more reliable results
- adding statistical comparisons
- tuning hyperparameters more systematically
- comparing with more advanced deep GNN methods

## Academic Note

This project was completed as part of my **3rd semester coursework** and is intended as an academic exploration of over-smoothing in deep GNNs.

## Author

**Dinesh Murthy Kadali**
