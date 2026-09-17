import torch

from custom_esm.models.esm3 import ESM3


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
L = len(sequence)

T_sftm = 1.0


# ------------------------------------------------------------
# Load ESM3
# ------------------------------------------------------------
print("Loading ESM3...")
model = ESM3.from_pretrained("esm3-open", device=device)
model.eval()

print("Embedding dtype:", model.encoder.sequence_embed.weight.dtype)
print("device:", device)
print("model device:", model.device)



# ------------------------------------------------------------
# Amino-acid mapping
# ------------------------------------------------------------
from custom_esm.utils.constants import esm3 as C

AA = "ACDEFGHIKLMNPQRSTVWY"

used_vocab = C.SEQUENCE_USED_VOCAB

print("SEQUENCE_USED_VOCAB:", used_vocab)
print("len:", len(used_vocab))

aa_to_idx = {
    aa: used_vocab.index(aa)
    for aa in AA
}

Z = torch.zeros(
    1,
    L,
    len(used_vocab),
    device=device,
    dtype=model.encoder.sequence_embed.weight.dtype,
    requires_grad=True,
)

with torch.no_grad():
    for i, aa in enumerate(sequence):
        Z[0, i, aa_to_idx[aa]] = 5.0



# ------------------------------------------------------------
# Continuous sequence probabilities
# ------------------------------------------------------------
P = torch.softmax(Z / T_sftm, dim=-1)

print("P.requires_grad:", P.requires_grad)
print("P.grad_fn:", P.grad_fn)


# ------------------------------------------------------------
# Prepare ESM3 default inputs
# ------------------------------------------------------------
default_tokens, affine, affine_mask = model._default(
    L + 2,
    device,
)


# ESM3 is loaded in bfloat16
default_tokens = list(default_tokens)
default_tokens[1] = default_tokens[1].to(torch.bfloat16)  # average_plddt
default_tokens[2] = default_tokens[2].to(torch.bfloat16)  # per_res_plddt
default_tokens = tuple(default_tokens)

affine = affine.to(dtype=torch.bfloat16)

# default_tokens contains:
# structure_tokens
# average_plddt
# per_res_plddt
# ss8_tokens
# sasa_tokens
# function_tokens
# residue_annotation_tokens


# ------------------------------------------------------------
# Continuous sequence -> ESM3 encoder
# ------------------------------------------------------------
x = model.encoder.custom_forward(
    P,
    *default_tokens,
)

print("\nEncoder output:")
print("  shape:", x.shape)
print("  requires_grad:", x.requires_grad)
print("  grad_fn:", x.grad_fn)


print("affine dtype:", affine.dtype)
print("affine rot dtype:", affine.rot.dtype)
print("affine trans dtype:", affine.trans.dtype)
print("affine_mask dtype:", affine_mask.dtype)
# ------------------------------------------------------------
# Transformer
# ------------------------------------------------------------
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    x, embedding, _ = model.transformer(
        x,
        sequence_id=None,
        affine=affine,
        affine_mask=affine_mask,
        chain_id=None,
    )

    structure_logits = model.output_heads.structure_head(x)

    

print("\nTransformer output:")
print("  shape:", x.shape)
print("  requires_grad:", x.requires_grad)
print("  grad_fn:", x.grad_fn)


# ------------------------------------------------------------
# Structure logits
# ------------------------------------------------------------
#structure_logits = model.output_heads.structure_head(x)

print("\nStructure logits:")
print("  shape:", structure_logits.shape)
print("  requires_grad:", structure_logits.requires_grad)
print("  grad_fn:", structure_logits.grad_fn)


# ------------------------------------------------------------
# Define a simple differentiable energy
# ------------------------------------------------------------
U = structure_logits.square().mean()

print("\nEnergy:")
print("  U =", U.item())
print("  requires_grad:", U.requires_grad)
print("  grad_fn:", U.grad_fn)


# ------------------------------------------------------------
# Backpropagate to sequence logits Z
# ------------------------------------------------------------
U.backward()

print("\nGradient:")
print("  Z.grad is None:", Z.grad is None)

if Z.grad is not None:
    print("  Z.grad shape:", Z.grad.shape)
    print("  Z.grad norm:", Z.grad.norm().item())
    print("  Z.grad min:", Z.grad.min().item())
    print("  Z.grad max:", Z.grad.max().item())
    print("  finite:", torch.isfinite(Z.grad).all().item())