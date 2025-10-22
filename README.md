# CognitiveMirrors
The official code of "Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning".

<img src="https://github.com/user-attachments/assets/3d45c7a8-c60f-4e17-9f23-882e008650ab" width="40">
Cognitive Mirrors investigates the functional diversity of attention heads in large language models (LLMs).

1. Environment Setup

2. Get Head Activations

Run python get_activations.py --mode train 

Run python get_activations.py --mode test

3. Probing

Run python get_topk_tokens_by_answer.py to get the activations of top-k meaningful tokens from generated answer.

Run python get_importance.py

4. Experiments

Run python ablation_topk.py to validate Functional Contributions of Cognitive Heads (Table 1 in paper).

Run python ablation_cot.py for Hierarchical structure exploration (Table 3 in paper).

Run python ablation_task_mask.py for Negative Intervention.

Run python ablation_task_recorrect.py for Positive Intervention.

