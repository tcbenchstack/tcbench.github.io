# import pytest
# import pathlib
# import hashlib

# from tcbench.modeling import utils
#
#
# def pytest_configure():
#     pytest.DIR_RESOURCES = (pathlib.Path(__file__).parent / "resources").resolve()
#
#
# @pytest.helpers.register
# def verify_deeplearning_model(fname, reference_fname, epsilon=None):
#     """Verifying trained model weights"""
#     import torch
#
#     net = torch.load(fname)
#     ref_net = torch.load(reference_fname)
#
#     assert len(net) == len(ref_net)
#     assert sorted(net.keys()) == sorted(ref_net.keys())
#
#     for name in net.keys():
#         weights = net[name]
#         ref_weights = ref_net[name]
#         if epsilon is None:
#             assert (weights.flatten() == ref_weights.flatten()).all()
#         else:
#             assert ((weights.flatten() - ref_weights.flatten()).abs() < epsilon).all()
#
#
# def _get_md5(fname):
#     data = pathlib.Path(fname).read_bytes()
#     md5 = hashlib.md5(data)
#     return md5.hexdigest()
#
#
# @pytest.helpers.register
# def verify_md5_model(fname, reference_fname):
#     assert _get_md5(fname) == _get_md5(reference_fname)
#
#
# @pytest.helpers.register
# def verify_reports(
#     folder, reference_folder, with_train=True, with_val=True, with_test=True
# ):
#     """Verify classification report and confusion matrixes"""
#     import pandas as pd
#
#     # note: by using folder / test*.csv automatically
#     # skips leftover if not found
#
#     def _add_file(folder, fname, fname_list):
#         if not (folder / fname).exists():
#             raise RuntimeError(f"missing {fname}")
#         fname_list.append(fname)
#
#     fnames = []
#     if with_train:
#         _add_file(folder, "train_class_rep.csv", fnames)
#         _add_file(folder, "train_conf_mtx.csv", fnames)
#     if with_val:
#         _add_file(folder, "val_class_rep.csv", fnames)
#         _add_file(folder, "val_conf_mtx.csv", fnames)
#     if with_test:
#         tmp = list(folder.glob("test*.csv"))
#         assert len(tmp) != 0
#         fnames.extend([item.name for item in tmp])
#
#     if len(fnames) == 0:
#         raise RuntimeError("empty list of files to verify")
#
#     for fname in fnames:
#         df = pd.read_csv(folder / fname)
#         ref_df = pd.read_csv(reference_folder / fname)
#         assert (df == ref_df).all().all()
#
#
# @pytest.helpers.register
# def match_run_hashes(folder, reference_folder, params_to_match=['seed', 'split_index', 'flowpic_dim', 'aug_name']):
#    
#     ref_catalog = {
#         path.name: utils.load_yaml(path / 'params.yml')
#         for path in reference_folder.iterdir()
#     }
#
#     pairs = []
#     for path in folder.iterdir():
#         curr_params = utils.load_yaml(path / 'params.yml')
#         curr_hash = path.name
#
#         curr_pair = [curr_hash, None]
#         for ref_hash, ref_params in ref_catalog.items():
#             tmp1 = {}
#             tmp2 = {}
#             for param_name in params_to_match:
#                 tmp1[param_name] = str(curr_params[param_name])
#                 tmp2[param_name] = str(ref_params[param_name])
#
#             if tmp1 == tmp2:
#                 curr_pair[-1] = ref_hash
#                 del(ref_catalog[ref_hash])
#                 break
#
#         pairs.append(curr_pair)
#
#     return pairs
