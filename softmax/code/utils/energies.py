import torch


def compute_U_cm(cm, ref_cm):
	return torch.triu((cm - ref_cm)**2.).sum()

def compute_U_am(am, ref_am):
	return torch.triu((torch.log(am) - torch.log(ref_am))**2.).sum()

#energia calcolata dai logits strutturali		#cross entropy
def compute_U_structure_ce(structure_logits, ref_structure_tokens):
    """
    Cross-entropy between the predicted structure-token distribution
    and the structure-token sequence of the reference structure.

    structure_logits:	
        [B, L, V]	#[1 proteina, lunghezza proteina, 4096 token di struttura]

    ref_structure_tokens:
        [B, L] integer structure-token IDs
    """

    B, L, V = structure_logits.shape				#assegno dimensioni

    return torch.nn.functional.cross_entropy(		#esegue softmax internamente: −log(softmax(x))
        structure_logits.reshape(B * L, V),			#trasformo [B, L, V] in [B*L, V] perche cross_entropy richiede [number_of_examples(siti), number_of_classes(tokens)]
        ref_structure_tokens.reshape(B * L),		#target [1, 58]-> [58] perche cross_entropy richiede [number_of_examples(siti)]
    )													#e.g. [42, 817, ..., 3901] ogni numero dice quale è token strutturale corretto per quella posizione




def compute_entropy(p, eps=1.0e-9):
	return (-p*torch.log(p+eps)).sum(axis=-1).sum()


""" ########################### """
""" ########## EXTRA ########## """
""" ########################### """

def compute_FoC(cm, ref_cm):
	DoC = 0
	for row in range(len(cm)-2):
		DoC += abs(cm[row, row+2:] - ref_cm[row, row+2:]).sum()
	norm = torch.sum(cm + ref_cm) - (3*len(cm)-2)*2
	return DoC / norm


