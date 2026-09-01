import torch


def compute_U_cm(cm, ref_cm):
	return torch.triu((cm - ref_cm)**2.).sum()

def compute_U_am(am, ref_am):
	return torch.triu((torch.log(am) - torch.log(ref_am))**2.).sum()

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
