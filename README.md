# HistoForge Models

Model architectures, training, and reference inference for HistoForge pathology
modules, built on Cellpose. The backend and QuPath extension are separate projects.

## Supported model architectures

| Architecture | Prediction head | Output |
| --- | --- | --- |
| DINOv3 ViT-L/16 + spatial adapter | Cellpose flows | Instance segmentation, with optional object classification |
| DINOv3 ViT-L/16 + spatial adapter | HoVer-Net horizontal/vertical distances | Instance segmentation, with optional object classification |
| DINOv3 ViT-L/16 + spatial adapter | Pixel classifier | Semantic segmentation (pixel classification) |

Model weights and datasets are not included. Each trained module has its own
evaluation results and usage limits.

## Documentation

- [HistoForge setup and workflows](https://histoforge.github.io/docs/)
- [Architecture and checkpoint format](prod_docs/architecture.md)
- [Standalone training and evaluation](prod_docs/training.md)
- [Shared components for managed training](prod_docs/managed_training.md)

## License

HistoForge modifications use Apache-2.0 and retain the upstream Cellpose
BSD-3-Clause notices. See [LICENSE](LICENSE) and [NOTICE](NOTICE). DINOv3 weights,
derived checkpoints, and datasets have separate terms.
