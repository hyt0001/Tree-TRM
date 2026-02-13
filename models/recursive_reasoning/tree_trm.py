from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TreeRecursiveReasoningModel_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TreeRecursiveReasoningModel_Carry:
    inner_carry: TreeRecursiveReasoningModel_InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TreeRecursiveReasoningModel_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int  # Number of tree depth steps (similar to H_cycles)
    L_cycles: int  # Number of internal processing steps per node
    
    # Tree specific
    tree_branching_factor: int = 4 # Number of child nodes per node (K)

    H_layers: int # ignored
    L_layers: int  # Layers in the reasoning module

    # Transformer config
    hidden_size: int  # Hidden size
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int  # ACT max steps
    halt_exploration_prob: float  # Q-learning exploration prob

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss

class TreeRecursiveReasoningModel_Block(nn.Module):
    def __init__(self, config: TreeRecursiveReasoningModel_Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            # Optional: MLP instead of Attention
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            # Standard Attention
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TreeRecursiveReasoningModel_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TreeRecursiveReasoningModel_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TreeRecursiveReasoningModel_Inner(nn.Module):
    def __init__(self, config: TreeRecursiveReasoningModel_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Module (Shared L-level)
        self.L_level = TreeRecursiveReasoningModel_ReasoningModule(layers=[TreeRecursiveReasoningModel_Block(self.config) for _i in range(self.config.L_layers)])

        # Tree Branching Embeddings: K learnable vectors representing different "child directions"
        # We add these to z before processing to simulate visiting different nodes
        self.branch_embeddings = nn.Parameter(
            trunc_normal_init_(torch.empty(self.config.tree_branching_factor, self.config.hidden_size), std=0.02)
        )
        
        # Chunk Encoder (for LongBench context chunks)
        # Assuming simple projection for now, but could be a small Transformer
        self.chunk_encoder = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)

        # Cross Attention Projections for Branch Selection
        self.ca_q_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.ca_k_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.ca_v_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.ca_o_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        
        # Branch Predictor: Projects context-aware state to K logits
        self.branch_pred_head = CastedLinear(self.config.hidden_size, self.config.tree_branching_factor, bias=False)


        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings (or Context Chunks)
        if puzzle_identifiers.ndim == 3: # [Batch, NumChunks, ChunkLen] -> Context Chunks Mode
            # 1. Embed tokens
            # [Batch, NumChunks, ChunkLen, Hidden]
            chunk_embeds = self.embed_tokens(puzzle_identifiers.to(torch.int32)) 
            
            # 2. Encode chunks into vectors
            # Simple Mean Pooling: [Batch, NumChunks, Hidden]
            chunk_vectors = chunk_embeds.mean(dim=2) 
            
            # 3. Project
            puzzle_embedding = self.chunk_encoder(chunk_vectors)
            
            # Note: In this mode, we don't concatenate puzzle_embedding to input embedding sequence directly
            # because puzzle_embedding is used as Key/Value Memory for Cross-Attention selection.
            # However, to keep compatibility with existing code structure that expects a combined sequence,
            # we might need to adjust.
            
            # Strategy: 
            # - Input Embedding is just the Query (Question).
            # - Puzzle Embedding is the Context Memory.
            # - The model expects `_input_embeddings` to return a single sequence.
            # - BUT, for Tree-TRM logic, we usually concat [Puzzle_Emb, Input_Emb].
            
            # Let's check `forward` method... it uses `input_embeddings` for L_level processing.
            # And `select_next_node` uses `z` which comes from L_level.
            
            # For LongBench RAG mode, we want:
            # - z (Latent State) to evolve.
            # - select_next_node to look at `puzzle_embedding` (Context Chunks) using `z` as query?
            # Wait, the current `select_next_node` uses `input_query` (from input) as Q, and `z` as K/V.
            
            # REVISION for RAG:
            # - We need `z` (State) to be Q.
            # - We need `Context Chunks` to be K/V.
            # - Then we select a chunk, and add it to `z`.
            
            # Let's strictly follow the existing interface but adapt usage.
            # We return `embedding` (Question) as the main sequence.
            # We store `puzzle_embedding` (Contexts) in a temporary way or just return it?
            # The current function signature returns only one tensor.
            
            # Hack: We attach puzzle_embedding to the module state temporarily? No, that's bad for DP.
            # Better: We assume this function returns the Question Embedding.
            # And we handle Context Chunks separately in forward.
            
            return self.embed_scale * embedding, puzzle_embedding # Return tuple! (Need to update call site)

        elif self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Fix: ensure puzzle embedding is reshaped correctly for concatenation
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
        else:
            # Fallback
            puzzle_embedding = torch.zeros(
                (input.shape[0], self.puzzle_emb_len, self.config.hidden_size),
                dtype=embedding.dtype,
                device=embedding.device
            )
        
        # Concatenate puzzle embedding (prefix) with token embedding
        embedding = torch.cat((puzzle_embedding, embedding), dim=1)  # Concat along sequence dim (dim=1)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding, None # Return tuple matching new signature

    def empty_carry(self, batch_size: int):
        return TreeRecursiveReasoningModel_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TreeRecursiveReasoningModel_InnerCarry):
        return TreeRecursiveReasoningModel_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def select_next_node(self, input_query: torch.Tensor, z: torch.Tensor, context_memory: Optional[torch.Tensor] = None):
        """
        Selects the best next 'child node' based on query relevance using Cross-Attention.
        input_query: [Batch, Hidden] - The pooled input serving as query (Q)
        z: [Batch, SeqLen, Hidden] - Current latent state 
        context_memory: [Batch, NumChunks, Hidden] - Optional external memory (for LongBench RAG)
        """
        batch_size = z.shape[0]
        
        # Determine Q, K, V source
        if context_memory is not None:
            # RAG Mode: 
            # Q = Current State Summary (z_pooled) or Input Query? 
            # We want to select a chunk relevant to current reasoning state AND original question.
            # Let's use (Query + State) as Q.
            z_pooled = z.mean(dim=1)
            q_input = input_query + z_pooled
            
            # K, V = Context Memory (Chunks)
            k_input = context_memory
            v_input = context_memory
            
            # We need to project context_memory to branch_embeddings space?
            # Or simpler: The "Output" of this selection is the selected Chunk Vector itself.
            pass
        else:
            # Standard Tree-TRM Mode
            q_input = input_query
            k_input = z
            v_input = z

        # 1. Project Q, K, V
        # Q: [Batch, 1, Hidden]
        q = self.ca_q_proj(q_input).unsqueeze(1)
        
        # K, V: [Batch, SeqLen/NumChunks, Hidden]
        k = self.ca_k_proj(k_input)
        v = self.ca_v_proj(v_input)
        
        # 2. Multi-head Cross Attention
        # Reshape for multi-head: [Batch, SeqLen, NumHeads, HeadDim]
        head_dim = self.config.hidden_size // self.config.num_heads
        
        q = q.view(batch_size, 1, self.config.num_heads, head_dim).transpose(1, 2) # [B, H, 1, D]
        k = k.view(batch_size, -1, self.config.num_heads, head_dim).transpose(1, 2) # [B, H, S, D]
        v = v.view(batch_size, -1, self.config.num_heads, head_dim).transpose(1, 2) # [B, H, S, D]
        
        # Scaled Dot Product Attention
        # Output: [B, H, 1, D]
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.config.hidden_size)
        
        # Output projection
        context_vector = self.ca_o_proj(attn_output).squeeze(1) # [Batch, Hidden]
        
        if context_memory is not None:
            # RAG Mode: The context_vector itself IS the retrieved information!
            # But to keep consistent with "branch embedding" logic (adding to z),
            # we can return it directly.
            # Also, we might want to know WHICH chunk was selected (for interpretability), 
            # but for now let's just use the soft-attended vector.
            selected_branch_emb = context_vector
        else:
            # 3. Predict best branch from context vector
            # logits: [Batch, K]
            logits = self.branch_pred_head(context_vector)
            
            # 4. Select best branch
            best_branch_idx = torch.argmax(logits, dim=-1) # [Batch]
            
            # 5. Get the corresponding branch embedding to add to z
            selected_branch_emb = self.branch_embeddings[best_branch_idx] # [Batch, Hidden]
        
        return selected_branch_emb

    def forward(self, carry: TreeRecursiveReasoningModel_InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TreeRecursiveReasoningModel_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input Embeddings
        # Note: _input_embeddings returns a tuple (embedding, context_memory)
        # where context_memory is only present in RAG mode (LongBench)
        input_embeddings, context_memory = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        # Compute Query from Input (Global Average Pooling) to serve as the fixed "User Query"
        # We exclude the puzzle_emb part for query if needed, or keep it. Let's keep it.
        # input_embeddings: [Batch, SeqLen, Hidden]
        query_vector = input_embeddings.mean(dim=1) # [Batch, Hidden]

        # Tree Traversal Loop
        # At each H-step, we select a "child node" (branch) relevant to the query
        
        z_H, z_L = carry.z_H, carry.z_L
        
        # We run H_cycles steps. Each step we move to a "child".
        # We treat this as updating z_H with a selected branch perturbation.
        
        # Optimization: TBPTT style (only last step has grad for H-loop) if configured, 
        # but TRM default config often has small H_cycles (3).
        
        # Loop Logic:
        # 1. Current z_H is the "parent".
        # 2. Select best branch based on (Query, z_H).
        # 3. Add branch embedding to z_H (conceptually moving to child).
        # 4. Process with L-level.
        
        grad_cycles = 1 # Last cycle has grad
        no_grad_cycles = self.config.H_cycles - grad_cycles
        
        # No Grad Phase
        with torch.no_grad():
            for _H_step in range(no_grad_cycles):
                # 1. Selection Phase
                branch_emb = self.select_next_node(query_vector, z_H, context_memory=context_memory) # [Batch, Hidden]
                
                # 2. Update z_H to "child" (add branch embedding broadcasting over SeqLen)
                z_H_child = z_H + branch_emb.unsqueeze(1)
                
                # 3. Process node (Reasoning)
                # Inner loop L
                for _L_step in range(self.config.L_cycles):
                     z_L = self.L_level(z_L, z_H_child + input_embeddings, **seq_info)
                z_H = self.L_level(z_H_child, z_L, **seq_info)

        # Grad Phase
        for _H_step in range(grad_cycles):
             # 1. Selection Phase
            branch_emb = self.select_next_node(query_vector, z_H, context_memory=context_memory)
            
            # 2. Update z_H to "child"
            z_H_child = z_H + branch_emb.unsqueeze(1)
            
            # 3. Process node
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H_child + input_embeddings, **seq_info)
            z_H = self.L_level(z_H_child, z_L, **seq_info)


        new_carry = TreeRecursiveReasoningModel_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TreeRecursiveReasoningModel(nn.Module):
    """Tree-TRM Wrapper (ACT compatible)"""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TreeRecursiveReasoningModel_Config(**config_dict)
        self.inner = TreeRecursiveReasoningModel_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TreeRecursiveReasoningModel_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TreeRecursiveReasoningModel_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TreeRecursiveReasoningModel_Carry, Dict[str, torch.Tensor]]:

        # Update carry
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Run Inner Model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Target Q
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TreeRecursiveReasoningModel_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

