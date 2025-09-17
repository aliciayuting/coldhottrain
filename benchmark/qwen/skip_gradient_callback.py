from transformers import TrainerCallback, PreTrainedModel
import torch.distributed as dist
import torch
import hashlib
import os

def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

#Q: would it be more efficient to calc the gradient mask on rank 0 and broadcast it to all ranks?
class SkipGradientCallback(TrainerCallback):
    def __init__(self,
                 model: PreTrainedModel,
                 epoch_start_track=1,
                 epoch_compute_masks=2,
                 zero_bottom_k_percent=0.1,
                 zero_mode="neurons",
                 freeze_params=False,
                 output_dir="",
                 ):
            self.model = model
            self.epoch_start_track = epoch_start_track
            self.epoch_compute_masks = epoch_compute_masks
            self.zero_bottom_k_percent = zero_bottom_k_percent
            self.zero_mode = zero_mode
            self.output_dir = output_dir

            self._has_computed_masks = False

            self.is_tracking_grads = False
            self.neuron_masks = {}
            # storage for split computation path
            self._neuron_scores = None         # List[Tensor] of per-param row L2 norms
            self._neuron_param_refs = None     # List[Tuple[name, rows]] matching scores
            self._total_neuron_rows = 0

    def _compute_masks(self):
        if self._has_computed_masks:
            return
        
        if _is_main():
            print(f"Computing skip-gradient masks at epoch {self.epoch_compute_masks}...")
        if self.zero_mode == "neurons":
            self.neuron_masks = self._build_neuron_masks_from_scores()
            # save masks to output_dir
            if self.output_dir and _is_main():
                os.makedirs(self.output_dir, exist_ok=True)
                cpu_masks = {k: v.detach().to("cpu", dtype=torch.bool) for k, v in self.neuron_masks.items()}
                out_path = os.path.join(self.output_dir, "neuron_masks.pt")
                torch.save(cpu_masks, out_path)
                print(f"[skipgradient] saved neuron masks to {out_path}")
            
        else:
            raise ValueError(f"Invalid zero_mode: {self.zero_mode}")
        
        self._has_computed_masks = True
    #Q: do we want to mask out bias grads too?
    @torch.no_grad()
    def _apply_fixed_masks(self):
        """Apply previously computed fixed masks to current gradients."""
        if self.zero_mode == "neurons":
            if not self.neuron_masks:
                raise RuntimeError("No neuron masks computed yet")
            for name, p in self.model.named_parameters():
                if p.grad is None or p.grad.ndim < 2:
                    continue
                row_mask = self.neuron_masks.get(name)
                if row_mask is not None:
                    # Zero out entire rows where row_mask is True
                    p.grad[row_mask] = 0.0
        else:
            raise ValueError(f"Unknown zero_mode: {self.zero_mode}")
        
    @torch.no_grad()
    def _collect_neuron_grad_norms(self):
        """Compute and store per-row L2 gradient norms for all 2D+ parameters."""
        scores = []
        param_refs = []  # (name, rows)
        for name, p in self.model.named_parameters():
            if p.grad is None or p.grad.ndim < 2:
                continue
            if p.grad.ndim > 2:
                print(f"[CustomTrainer] param {name} has ndim={p.grad.ndim}")
            g = p.grad.detach()
            rows = g.shape[0]
            flat = g.view(rows, -1)
            l2 = (flat.pow(2).sum(dim=1)).float()  # [rows]
            scores.append(l2)
            param_refs.append((name, rows))

        self._neuron_scores = scores
        self._neuron_param_refs = param_refs
        self._total_neuron_rows = int(sum(t.numel() for t in scores)) if scores else 0
        return scores, param_refs

    @torch.no_grad()
    def _build_neuron_masks_from_scores(self):
        """Build boolean row masks per parameter using stored per-row gradient norms."""
        scores = self._neuron_scores or []
        param_refs = self._neuron_param_refs or []
        if not scores:
            return {}

        all_scores = torch.cat(scores)
        total_rows = all_scores.numel()
        k = int(total_rows * self.zero_bottom_k_percent)
        if k <= 0:
            return {}

        # Bottom-K indices across all rows of all relevant params (stable for ties)
        order = torch.argsort(all_scores, stable=True)
        bottom_idx = order[:k]

        # Build boolean row masks per param
        masks = {}
        offset = 0
        for (name, rows), l2 in zip(param_refs, scores):
            this_slice = bottom_idx[(bottom_idx >= offset) & (bottom_idx < offset + rows)] - offset
            if this_slice.numel() > 0:
                row_mask = torch.zeros(rows, dtype=torch.bool, device=l2.device)
                row_mask[this_slice] = True
                masks[name] = row_mask
            offset += rows
        return masks

    def on_train_begin(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[skipgradient][on_train_begin] max_steps={state.max_steps} epochs={args.num_train_epochs}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[skipgradient][on_epoch_begin] epoch_float={state.epoch}")
        if int(state.epoch) >= self.epoch_start_track and not self._has_computed_masks: # type: ignore
            self.is_tracking_grads = True
        if int(state.epoch) == self.epoch_compute_masks and not self._has_computed_masks: # type: ignore
            self._compute_masks()
            self._ddp_assert_neuron_masks_identical()
            self.is_tracking_grads = False  # stop tracking after computing masks

    def on_train_batch_end(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[skipgradient][on_train_batch_end] global_step={state.global_step}")
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            print("no optimizer :(")

        if(optimizer):
            print(optimizer)
        #if _is_main():
            #print(f"[skipgradient][on_pre_optimizer_step] global_step={state.global_step}")

        if self.is_tracking_grads and not self._has_computed_masks:
            self._collect_neuron_grad_norms()
        elif self._has_computed_masks:
            #print(f"[skipgradient][on_pre_optimizer_step] global_step={state.global_step} applying fixed masks")
            self._apply_fixed_masks()



    # def on_step_end(self, args, state, control, **kwargs):
    #     # Fires after optimizer step; here global_step has just incremented
    #     if _is_main():
    #         print(f"[skipgradient][on_step_end] global_step={state.global_step}")
            
    # def on_substep_end(self, args, state, control, **kwargs):
    #     # Fires after optimizer step; here global_step has just incremented
    #     if _is_main():
    #         print(f"[skipgradient][on_substep_end] global_step={state.global_step}")


    @torch.no_grad()
    def _ddp_assert_neuron_masks_identical(self):
        """Check that self._fixed_neuron_masks are identical across ranks.
        Call only on a synchronized step (require_backward_grad_sync == True)."""
        zeroed = sum(int(m.sum().item()) for m in self.neuron_masks.values()) if self.neuron_masks else 0
        print(f"[skipgradient] masks: zeroing {zeroed}/{self._total_neuron_rows} elements (~{(zeroed/max(1,self._total_neuron_rows))*100:.2f}%).")
        if not (dist.is_available() and dist.is_initialized()):
            return  # single GPU or non-DDP


        world_size = dist.get_world_size()
        print(f"[MaskCheck] Verifying neuron masks across {world_size} ranks...")
        rank = dist.get_rank()

        # Canonical param list: all 2D+ params in named order
        names = []
        shapes = []
        for name, p in self.model.named_parameters():
            if p.ndim >= 2:
                names.append(name)
                shapes.append(p.shape)

        # Build a vector of int64 checksums, one per param in 'names'
        checks = []
        for (name, shape) in zip(names, shapes):
            rows = shape[0]
            mask = self.neuron_masks.get(name, None)
            if mask is None:
                # Treat as all-zeros (no rows selected)
                mask_cpu = torch.zeros(rows, dtype=torch.bool).cpu()
            else:
                # Ensure correct length and on CPU
                assert mask.ndim == 1 and mask.numel() == rows, f"Row mask shape mismatch for {name}"
                mask_cpu = mask.detach().to("cpu", dtype=torch.bool)

            # Hash the raw bytes for exact equality (name+shape+mask contents)
            h = hashlib.sha256()
            h.update(name.encode("utf-8"))
            h.update(str(tuple(shape)).encode("utf-8"))
            h.update(mask_cpu.numpy().tobytes())
            # take first 8 bytes as unsigned 64-bit int
            h64 = int.from_bytes(h.digest()[:8], byteorder="big", signed=True)
            checks.append(h64)

        local_vec = torch.tensor(checks, dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")

        # Broadcast rank-0's vector as the reference
        ref_vec = local_vec.clone()
        dist.broadcast(ref_vec, src=0)

        # Compare to reference and report any mismatches
        mism = (local_vec != ref_vec)
        num_mism = int(mism.sum().item())

        # Aggregate across all ranks to see if any mismatches anywhere
        total_mism = torch.tensor([num_mism], dtype=torch.int32, device=local_vec.device)
        dist.all_reduce(total_mism, op=dist.ReduceOp.SUM)

        if total_mism.item() != 0:
            # Print which params differ on this rank (keep it concise)
            bad_idxs = mism.nonzero(as_tuple=False).flatten().tolist()
            bad_names = [names[i] for i in bad_idxs[:100]]  # truncate to first 100 for readability
            print(f"[MaskCheck][rank {rank}] {num_mism} param masks differ from rank 0. e.g., {bad_names}")
            # Optional: hard assert to fail fast
            # raise RuntimeError(f"DDP mask mismatch on rank {rank}")
        elif rank == 0:
            print("[MaskCheck] All ranks have identical neuron masks ✔")
