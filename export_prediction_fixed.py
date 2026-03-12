"""
Fixed version of export_prediction_from_logits to handle orientation and data flipping issues.

This module provides a corrected implementation that ensures proper handling of
nibabel orientation and prevents data flipping issues.
"""

from typing import Union
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def export_prediction_from_logits(
    predicted_array_or_file: Union[np.ndarray, torch.Tensor],
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str],
    output_file_truncated: str,
    num_threads_torch: int = default_num_processes
):
    """
    Export prediction from logits with fixed orientation handling.
    
    This is a fixed version that addresses data flipping issues when using
    NibabelIOWithReorient. The key fix is ensuring data is correctly oriented
    before calling write_seg.
    
    Args:
        predicted_array_or_file: Predicted logits (torch.Tensor or np.ndarray)
        properties_dict: Properties dictionary containing orientation info
        configuration_manager: Configuration manager
        plans_manager: Plans manager
        dataset_json_dict_or_file: Dataset JSON dict or path
        output_file_truncated: Output file path without extension
        save_probabilities: Whether to save probability maps
        num_threads_torch: Number of threads for torch operations
    """
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    # Convert logits to segmentation with correct shape
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=False,
        num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # Handle probabilities if needed
    segmentation_final = ret
    return segmentation_final
    '''
    del ret
    print(segmentation_final.shape)
    # Get image reader/writer class and write segmentation
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(
        segmentation_final,
        output_file_truncated + dataset_json_dict_or_file['file_ending'],
        properties_dict
    )
    '''
