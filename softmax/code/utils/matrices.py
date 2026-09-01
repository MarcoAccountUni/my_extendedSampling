import torch


def compute_contact_map(coordinates, d0):
	distmatrix = compute_distance_matrix(coordinates, len(coordinates))
	return (distmatrix <= d0).int()

def compute_distance_matrix(coordinates, L):
	distance_matrix = torch.zeros((L, L))
	masks = ~torch.isinf(coordinates[..., 0])
	for row in range(L):
		for col in range(row+1, L):
			distance_matrix[row, col] = compute_residue_distance(
				coordinates[row, masks[row]], 
				coordinates[col, masks[col]]
			)
			distance_matrix[col, row] = distance_matrix[row, col]
	return distance_matrix

def compute_residue_distance(residue_one, residue_two):
	distances = []
	for xyz_one in residue_one:
		for xyz_two in residue_two:
			distance = torch.sqrt( ((xyz_one - xyz_two)**2.).sum() )
			distances.append(distance.item())
	return min(distances)
