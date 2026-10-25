"""Pin the upstream-owned functions used by the HistoForge runtime."""

import ast
import copy
import hashlib
from pathlib import Path


UPSTREAM_CELLPOSE_COMMIT = "a54cb48849b7e225a81e8e43dcb042d42427f543"

# SHA-256 of the executable function AST at the commit above. This excludes
# docstrings, formatting and line positions while rejecting semantic edits.
UPSTREAM_FUNCTION_AST_SHA256 = {
    "histoforge_models/core.py": {
        "_forward": "41e0ec5b3e360b67bc6213a15a34e996dfdfd11d585241cffc08f11e81f01337",
    },
    "histoforge_models/dynamics.py": {
        "_extend_centers_gpu": "f88f4bea46fd0b3b6fa36600e773cce5e9854632904101e3a1317b54c4d09255",
        "center_of_mass": "a07a1a154ffe0c3fe0fbda92e3b662ee5cabf70cc29fdf564f40adbeec84314a",
        "get_centers": "811e35bef8a0bb11235f8b1ca4a88aee0b72edd1e3069971e90de5f452b7248c",
        "masks_to_flows_gpu": "5d573854567afd36255fddb59cb8eac139657773df085125e53c1487c5c36ee4",
        "masks_to_flows_gpu_3d": "6126e705dc7a2518e59301e743994c1f6235544a91d30a163ba1327506319e44",
        "labels_to_flows": "6e11b29bda639e66d57967d183b344f40afd3eb3fb82f506c5f0429111e474a5",
        "flow_error": "15aef607c29074801cc1eac8dd2953202e6f476097efd1358df4bc75eaa04de9",
        "steps_interp": "365b9c403904606a15376c4985495e9b0ee5c7e639cd190d2c6cb93d78a84361",
        "follow_flows": "045b8a7205c8b8bbaea5928be11b491a1991dc0a6b20dbb889203e8516e3ab31",
        "remove_bad_flow_masks": "59dc5838184df811dd70c03767dfafff07e5798fde69fa972f7ec03555bb0011",
        "max_pool1d": "04d6cacab20c753de43a9eb880d2bf4b4a62f9eb5046d2c803635dd28927b1c7",
        "max_pool_nd": "f9b034fa19aca82a150879c65bb3bd5374b4a420763dca06422f2d73c98ea915",
        "get_masks_torch": "4aca20db575780f6059f27216c7584efbbfd224bdf9f45bc415fff5b39787ee5",
        "resize_and_compute_masks": "2fb78ca7606e68a46df17c7f5070eafbe1ab76dca50a16824fbc5fb0aab4f16c",
        "compute_masks": "ce0f4ef2f6e35593668f991da1c1d546b7e816fcd5bfbda229858e0a3bb31fe4",
    },
    "histoforge_models/transforms.py": {
        "_taper_mask": "84ca5ed137bf2a1ed1fc7482833c456916d1c0d7bc3d980c1a91fd394ffcd17d",
        "average_tiles": "661ccf7e78f651df498977dd812fcae280a0f1bdedb2428157dc17d750af0f59",
        "make_tiles": "383c11ea825091c1c464ae9b013b57b939124bd06f6919a17ea69d0a3950fe3e",
        "get_pad_yx": "df6e76fa270feaad2073c94acafb39c2484cc9475056db276e33c14da329a973",
    },
    "histoforge_models/utils.py": {
        "fill_holes_and_remove_small_masks": "5ca9687479e33bd99906938c245ab8267ddc02b50ff3a3b233fa4eb631b60844",
    },
    "histoforge_models/metrics.py": {
        "mask_ious": "823e197b6448465122aa7e1a28b91e52f394a0d0cede8109561252c0536a0b03",
        "boundary_scores": "e1ee1f7f26e9eb6fc5af59e651247b41863b5b581750fdf9fd487372640e84c0",
        "_label_overlap": "aae90caa55c85434d140b921fe31bcffd49682161875b37b75f16767b54ba76e",
        "aggregated_jaccard_index": "7f319a0c52decb9b028296a13d68d9841bc8d54ba8f6751f07504ce657f83adb",
        "average_precision": "4333c24e8396b59a4ddc02611c111f3a4e08d13425c87bfc6a5c600691a022cd",
        "_intersection_over_union": "f21d5a2e5039e91bb08f9286dd4111f868a42d62bd76f0ffa3b36b5b8732383a",
        "_true_positive": "a7faa4e2d2b365eef741f25e1ae2f0459de4ca9bd2b3d39159ee99140b421ab4",
    },
    "histoforge_models/pathology/managed_training/_flow_targets.py": {
        "_extend_centers_gpu": "f88f4bea46fd0b3b6fa36600e773cce5e9854632904101e3a1317b54c4d09255",
        "center_of_mass": "a07a1a154ffe0c3fe0fbda92e3b662ee5cabf70cc29fdf564f40adbeec84314a",
        "get_centers": "811e35bef8a0bb11235f8b1ca4a88aee0b72edd1e3069971e90de5f452b7248c",
        "masks_to_flows_gpu": "5d573854567afd36255fddb59cb8eac139657773df085125e53c1487c5c36ee4",
    },
}


class _WithoutDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def _executable_ast(function):
    function = _WithoutDocstrings().visit(copy.deepcopy(function))
    return ast.dump(function, include_attributes=False)


def test_upstream_owned_function_asts_remain_pinned():
    repository_root = Path(__file__).resolve().parents[1]
    for relative_path, expected_functions in UPSTREAM_FUNCTION_AST_SHA256.items():
        tree = ast.parse((repository_root / relative_path).read_text())
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for function_name, expected_digest in expected_functions.items():
            payload = _executable_ast(functions[function_name])
            actual_digest = hashlib.sha256(payload.encode()).hexdigest()
            assert actual_digest == expected_digest, (
                f"{relative_path}:{function_name} differs from upstream Cellpose "
                f"{UPSTREAM_CELLPOSE_COMMIT}"
            )
