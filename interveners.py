import torch
import torch.nn as nn
import pyvene as pv
from typing import List, Optional, Tuple, Union, Dict, Any
from pyvene import IntervenableModel, CollectIntervention

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b

    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b
    
class head_Intervener:
    def __init__(self, head_mask):
        self.head_mask = head_mask  # [num_layers, num_heads] 的 mask 矩阵
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        """
        b: (batch, seq_len, num_heads * head_dim)
        屏蔽指定 layer 中为 0 的 head（只在指定 layer 执行）
        """
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.head_mask.to(b.device)
        self.actions.append(action.detach().clone())
        # print('1')
        b[0, -1] = b[0, -1] * action
        return b
    
# class head_Intervener:
#     def __init__(self, head_mask):
#         """
#         Args:
#             head_mask (torch.Tensor): shape = (num_layers, num_heads)，
#                                        里面是0/1，控制每个head是否保留。
#         """
#         self.head_mask = head_mask  # (num_layers, num_heads)

#     def __call__(self, x, layer_idx=None):
#         """
#         干预函数：对 attention 的输出 x 进行处理。

#         Args:
#             x (torch.Tensor): 输入的 attention heads 输出，shape = (batch_size, seq_len, hidden_dim)
#             layer_idx (int, optional): 当前层数，告诉我们应用哪个layer的mask。

#         Returns:
#             torch.Tensor: 干预后的输出
#         """
#         # 先确定 x 的 shape
#         batch_size, seq_len, hidden_dim = x.shape

#         # 根据 hidden_dim 推算 head 数
#         num_heads = self.head_mask.shape[1]
#         head_dim = hidden_dim // num_heads

#         # Reshape: (batch_size, seq_len, num_heads, head_dim)
#         x = x.view(batch_size, seq_len, num_heads, head_dim)

#         # 取当前层的 mask (1, 1, num_heads, 1) for broadcasting
#         layer_head_mask = self.head_mask[layer_idx].view(1, 1, num_heads, 1)

#         # 应用 mask
#         x = x * layer_head_mask

#         # 再还原回原 shape
#         x = x.view(batch_size, seq_len, hidden_dim)

#         return x    

def get_batch_size(model_input):
    """
    Get batch size based on the input
    """
    if isinstance(model_input, torch.Tensor):
        batch_size = model_input.shape[0]
    else:
        for _, v in model_input.items():
            batch_size = v.shape[0]
            break
    return batch_size

class CustomGeneratedModel(IntervenableModel):
    def __init__(self, config, model):
        # 初始化模型
        super().__init__(config, model)

    def generate(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        intervene_on_prompt: bool = False,
        subspaces: Optional[List] = None,
        output_original_output: Optional[bool] = False,
        **kwargs,
    ):
        """
        Intervenable generation function that serves a
        wrapper to regular model generate calls.

        Currently, we support basic interventions **in the
        prompt only**. We will support generation interventions
        in the next release.

        TODO: Unroll sources and intervene in the generation step.

        Parameters:
        base:                The base example.
        sources:             A list of source examples.
        unit_locations:      The intervention locations of
                             base.
        activations_sources: A list of representations.
        intervene_on_prompt: Whether only intervene on prompt.
        **kwargs:            All other generation parameters.

        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.
        """
        # TODO: forgive me now, i will change this later.
        activations_sources = source_representations
        if sources is not None and not isinstance(sources, list):
            sources = [sources]
            
        self._cleanup_states()

        self._intervene_on_prompt = intervene_on_prompt
        self._is_generation = True
        
        if not intervene_on_prompt and unit_locations is None:
            # that means, we intervene on every generated tokens!
            unit_locations = {"base": 0}
        
        # broadcast
        unit_locations = self._broadcast_unit_locations(get_batch_size(base), unit_locations)
        sources = [None]*len(self._intervention_group) if sources is None else sources
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(activations_sources)
        subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)
        
        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )
        
        base_outputs = None
        if output_original_output:
            # returning un-intervened output
            base_outputs = self.model.generate(**base, **kwargs)

        set_handlers_to_remove = None
        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            
            # run intervened generate
            counterfactual_outputs = self.model.generate(
                **base, **kwargs
            )
            
            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(
                        self.interventions[key],
                        CollectIntervention
                    ):
                        collected_activations += self.activations[key]
        except Exception as e:
            raise e
        finally:
            if set_handlers_to_remove is not None:
                set_handlers_to_remove.remove()
            self._is_generation = False
            self._cleanup_states(
                skip_activation_gc = \
                    (sources is None and activations_sources is not None) or \
                    self.return_collect_activations
            )
        
        if self.return_collect_activations:
            return (base_outputs, collected_activations), counterfactual_outputs
        
        return base_outputs, counterfactual_outputs