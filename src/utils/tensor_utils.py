import os
import pickle
from typing import Optional, Tuple, Union

import torch

from src.utils.file_utils import open_local_or_remote


def locations_to_index_tuple(locations: torch.Tensor, num_dims: int = 2) -> Tuple:
    """
    Convert a tensor of locations to a tuple of index tensors for advanced indexing.

    Args:
        locations (torch.Tensor): A tensor of shape `[L, D]` where `L` is the number of
            locations and `D >= num_dims`.
        num_dims (int): The number of dimensions to extract. The first num_dims columns of
            the locations tensor are used. We explicitly specify this to make the
            function call traceable.

    Returns:
        Tuple: A tuple of `num_dims` tensors, each of shape `[L]` representing the
            indices for one dimension.

    Example:
        >>> locations = torch.tensor([[0, 10], [1, 20], [2, 5]])
        >>> locations_to_index_tuple(locations, num_dims=2)
        (tensor([0, 1, 2]), tensor([10, 20,  5]))

        >>> locations = torch.tensor([[0, 10], [1, 20], [2, 5]])
        >>> locations_to_index_tuple(locations, num_dims=1)
        (tensor([0, 1, 2]))
    """
    return tuple(locations[:, i] for i in range(num_dims))


def extract_locations(
    data: torch.tensor, locations: torch.tensor, num_dims: int = 2
) -> torch.tensor:
    """
    Extracts the elements from a tensor at the specified indices.

    Args:
        data (torch.tensor): The input tensor of N dimensions from which to extract elements.
        locations (torch.tensor): Tensor of shape [L, D] where L is the number of
        elements where each D dimensional row reprecents the first D dimensions
        of the data tensor to extract.
        num_dims (int): The number of dimensions to extract. The first num_dims columns of
        the locations tensors are used. We need to specify to make this function call traceable.

    Returns:
        torch.tensor: A tensor of shape [L,...] with total N-num_dims+1 dimensions
        containing the extracted elements.

    Example:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> locations = torch.tensor([[0, 1], [1, 2]])
        >>> extract_locations(data, locations, num_dims=2)
        tensor([2, 6]) # (Index 0,1 gives 2 (First row, second column); Index 1,2 gives 6 (Second row, third column))

        >>> locations = torch.tensor([[0, 1], [2, 0]])
        >>> extract_locations(data, locations, num_dims=1)
        tensor([[1, 2, 3], [7, 8, 9]])
        # (num_dims = 1 implies we are extracting based on the first dimension only.
        # Thus, we get the first row (from [0,1] as 1 is ignored) and the third row
        # (from [2,0] as 0 is ignored) of the data tensor.
    """

    # Separate the locations for each of the first D dimensions
    index_tuple = locations_to_index_tuple(locations=locations, num_dims=num_dims)

    # Use indexing with a tuple of index tensors
    extracted_values = data[index_tuple]

    return extracted_values


def merge_list_of_keyed_tensors_to_single_tensor(
    data: list[dict[str, torch.Tensor]],
    index_key: str,
    value_key: str,
) -> torch.Tensor:
    """
    Converts a list of dictionaries of id to tensors into a single tensor by squeezing
    the tensors along the specified index key.
    e.g.,
    data = [
    [
        {'user_id': 123,
        {'semantic_id': torch.tensor([21, 32, 124]),
        other features.....}
        ],
    [
        {'user_id': 456,
        {'semantic_id': torch.tensor([11, 22, 33]),
        other features.....}
    ]
    output:
    tensor([...,
            [21, 32, 124], # row 123
            ...,
            [11, 22, 33], # row 456
            ...
            ])

    Args:
        data (list[dict[str, torch.Tensor]]): A list of dictionaries where each dictionary
            contains an index key and a value key.
        index_key (str): The key in the dictionary that contains the index for each row.
        value_key (str): The key in the dictionary that contains the tensor to be merged.
    """
    batch_size = len(data)
    dimensions = torch.tensor(data[0][value_key]).size()
    output_tensor = torch.zeros((batch_size, *dimensions))
    for row in data:
        index = row[index_key]
        value = row[value_key]
        if index < batch_size:
            output_tensor[index] = torch.tensor(value)
        else:
            raise IndexError(
                f"Index {index} out of bounds for batch size {batch_size}."
            )
    return output_tensor


def deduplicate_rows_in_tensor(
    file_path: Optional[str] = None, return_tensor: bool = False
) -> Union[None, torch.Tensor]:
    """
    Identifies and de-duplicate repeated rows in a PyTorch tensor.
    Rows that are not duplicated will have a new column with value 0,
    while rows that are duplicated will have a new column indicating the number of duplicates from 1 to N-1
    where N is the number of duplicates for that row.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
        return_tensor: If True, returns the modified tensor; otherwise, saves it to the file.
    Returns:
        If return_tensor is True, returns the modified tensor with a new column indicating
        the number of duplicates for each row. If False, saves the modified tensor to the file.
    """
    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))
    assert len(data.size()) == 2, "Input data must be a 2D PyTorch tensor."

    # Use torch.unique to get unique rows and their inverse indices
    unique_rows, inverse_indices, counts = torch.unique(
        data, dim=0, return_inverse=True, return_counts=True
    )

    output_indices = torch.zeros_like(inverse_indices)

    # Find indices where counts > 1 (meaning duplicates exist)
    duplicate_indices = torch.where(counts > 1)[0]

    for i in range(len(duplicate_indices)):
        # Calculate number of collisions
        num_of_collisions = counts[duplicate_indices[i]]

        # Gather the indices where the collisions occur
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]

        # Create a range based on the number of collision, starting from 1
        range_to_add = torch.arange(1, num_of_collisions + 1)

        # Scatter to those specific indices
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)

    # Concatenate the duplicate indicator column to the original data
    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    if return_tensor:
        return result
    else:
        # Save the result to a file
        torch.save(result, file_path)
        return None


def transpose_tensor_from_file(
    file_path: Optional[str] = None,
    return_tensor: bool = False,
    dim1: int = -2,
    dim2: int = -1,
) -> Union[None, torch.Tensor]:
    """
    Transposes a PyTorch tensor from a file accoridng to designated dimensions.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
        return_tensor: If True, returns the modified tensor; otherwise, saves it to the file.
        dim1: The first dimension to transpose (default: -2).
        dim2: The second dimension to transpose (default: -1).
    Returns:
        If return_tensor is True, returns the modified tensor. If False, saves the modified tensor to the file.
    """
    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))

    # Transpose the tensor
    result = data.transpose(dim1, dim2)
    if return_tensor:
        return result
    else:
        # Save the result to a file
        torch.save(result, file_path)
        return None


def build_sid_tensor_from_pkl(
    file_path: Optional[str] = None,
    item_id_key: str = "item_id",
    cluster_ids_key: str = "cluster_ids",
) -> None:
    """Convert a merged_predictions.pkl (list of {item_id, cluster_ids} dicts) into a
    PyTorch tensor of shape (num_hierarchies, max_item_id+1), saved alongside the pkl as
    semantic_ids.pt.

    The tensor is indexed directly by item_id, so id_map.t()[item_id] returns the SID
    tuple for any item — regardless of whether item IDs are 0-indexed or sparse.

    Args:
        file_path: Path to merged_predictions.pkl produced by LocalPickleWriter.
        item_id_key: Dict key for the item ID (default: "item_id").
        cluster_ids_key: Dict key for the cluster ID tuple (default: "cluster_ids").
    """
    if file_path is None or not file_path.endswith("merged_predictions.pkl"):
        return None

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        return None

    max_id = max(int(row[item_id_key]) for row in data)
    num_hierarchies = len(data[0][cluster_ids_key])

    # Shape (max_id+1, num_hierarchies); rows for item IDs not in the map stay zero.
    tensor = torch.zeros((max_id + 1, num_hierarchies), dtype=torch.long)
    for row in data:
        iid = int(row[item_id_key])
        tensor[iid] = torch.as_tensor(row[cluster_ids_key], dtype=torch.long)

    # Transpose to (num_hierarchies, max_id+1) — the shape map_sparse_id_to_semantic_id expects.
    out_path = os.path.join(os.path.dirname(file_path), "semantic_ids.pt")
    torch.save(tensor.t().contiguous(), out_path)
    return None
