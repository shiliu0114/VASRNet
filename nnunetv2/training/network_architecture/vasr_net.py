"""VASR-Net architecture exposed from the project comparison module.

The implementation lives in ``dynamic_network_architectures_compare`` because
that module is also used by the parameter/FLOPs audit.  Re-exporting it here
gives training and checkpoint restoration one stable import path without
duplicating the network source.
"""

from dynamic_network_architectures_compare.architectures.vasr_net import VASRNet

__all__ = ["VASRNet"]
