# Custom ESM codes

This repository contains the customized code off the ESM models, both ESM2 (in `custom_fair_esm/`) and ESM3 (in `custom_esm/`).<br>
In order to be used, they must be copied within your python environment, alongside your other libraries.<br>

---
## Usage
Since ESM3 is the most recent model, we are going to focus on that one. Once you have downloaded this repository and copied `custom_esm/` into your personal python environment, you can start by initializing your model using the following snippet of code.
```python
from huggingface_hub import login
from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESM3InferenceClient

# The next line is necessary only for the first login. You can comment it afterwards.
login()
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open")
for param in model.parameters():
    param.requires_grad = False
```

If you want to check the usage of original ESM3 methods (as `model.generate()` or `model.decode()`), I suggest you look at the ESM3 repository on github.<br>
Let's focus on the main additional custom methods.
- `model.predict_attention()`: it takes as input `sequence_probs`, an LxV (with L as length of the sequence and V as the vocabulary dimension) "probability" matrix, that is, positive and normalized row-wise. The vocabulary contains the amino acids that we are studying during the exploration, that is it defines all the possible amino acid combinations encountered during the sampling. The custom list of studied amino acids is indicated as `SEQUENCE_USED_VOCAB` in `custom_esm/utils/contants/esm3.py`. The output is a symmetric attention matrix, which encodes the structural properties of the input combination of amino acids.
- `model.predict_protein()`: it takes as input an `ESMProteinTensor` (see ESM3 github repository), which contains a "tokenized" sequence of amino acids. Each token represents the index of the amino acid within the vocabulary. Let's say the vocabulary were `vocab=['A', 'B', 'C']`, then the sequence `seq=['A', 'A', 'C', 'B', 'C']` would be tokenized as `tok=[0, 0, 2, 1, 2]`. The other mandatory input variable is `config`. In short, it contains all the instructions necessary to ESM3 in order to understand how you want the structure to be produced. I suggest you use the predetermined `config` given by the function `init_structure_config` that you can find in the file `softmax/code/utils/predictions.py` in this repository. Finally, he output of this function will be an `ESMProtein`, with two important arguments: `coordinates`, conatining the coordinates of all the atoms along the input sequence, and `plddt`, that is, the confidence index for the predicted position of each atom.
<br>
**Suggestion**: instead of using `model.predict_protein()`, you could use the function `predict_structure()` in the file `softmax/code/utils/predictions.py`, which instead of an `ESMProtein` output gives back the contact map (given the predicted coordinates) and the plddt.

