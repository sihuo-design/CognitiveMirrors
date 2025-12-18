# Cognitive Mirrors

**Cognitive Mirrors** is the official codebase for **“Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning.”**

<p align="center">
  <img src="assets/test.png" width="700">
</p>

Cognitive Mirrors investigates the **functional diversity of attention heads** in large language models (LLMs).

---

## Environment Setup

Create and activate the environment (example with Conda):

```bash
conda create -n cognitive_mirrors python=3.8
conda activate cognitive_mirrors
pip install -r requirements.txt
```

## Get Head Activations

```bash
python get_activations.py --mode train 

python get_activations.py --mode test
```

## Probing

```bash
python get_topk_tokens_by_answer.py
```
to get the activations of top-k meaningful tokens from generated answer.

Compute head importance
```bash
python get_importance.py
```

## Experiments

**Functional Contributions of Cognitive Heads (Table 1)**

```bash
python ablation_topk.py
```

**Hierarchical Structure Exploration (Table 3)**

```bash
python ablation_cot.py
```

**Negative Intervention**

```bash
python ablation_task_mask.py
```

**Positive Intervention**

```bash
python ablation_task_recorrect.py
```
