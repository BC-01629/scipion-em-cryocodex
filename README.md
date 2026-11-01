# CryoCodex-Enhance for Scipion

A Scipion plugin for [CryoCodex](https://github.com/YangLab-SDU/CryoCodex), providing
cryo-EM density-map enhancement, local quality estimation and molecular mask prediction.

Available on [PyPI](https://pypi.org/project/scipion-em-cryocodex/).
Supports CryoCodex v1.0.0; tested with Scipion 3.8.3 on Linux.

## Requirements

- Scipion (`scipion-em >= 3.8`).
- An NVIDIA GPU with a CUDA-compatible driver.
- Conda, `wget` and network access for backend installation.

## Installation

Use your Scipion launcher to install the plugin and backend:

```bash
scipion3 python -m pip install scipion-em-cryocodex==0.3.2
scipion3 installb cryocodex-1.0.0
```

The backend installer creates a separate Python 3.11 environment and downloads the
model automatically. Restart Scipion after installation.

To upgrade an existing plugin:

```bash
scipion3 python -m pip install --upgrade scipion-em-cryocodex
```

### Using an existing CryoCodex installation

Skip backend installation and set the following paths in the `[PLUGINS]` section
of your Scipion `scipion.conf`:

```ini
[PLUGINS]
CRYOCODEX_HOME = /path/to/CryoCodex
CRYOCODEX_PYTHON = /path/to/cryocodex-env/bin/python
CRYOCODEX_MODEL = /path/to/CryoCodex.pth
```

## Usage

In the **Protocols SPA** view, select **3D → Postprocess → CryoCodex-Enhance**.

1. Select an imported volume.
2. Choose **Fast** to crop background or **Normal** to process the whole map.
3. Optionally use **Check crop** in Fast mode to review the crop before execution.
4. Set the GPU and processing options, then execute.

**Restore full map extent** fills cropped background with zeros.
**Return to input voxel size** restores input sampling; otherwise outputs use the
backend's 1 Å working grid.

The protocol produces an enhanced density map, local quality maps for the input
and enhanced maps, and a molecular mask. The Summary contains result paths and
ChimeraX coloring instructions.

## Verification

```bash
scipion3 inspect cryocodex
scipion3 test cryocodex.tests.test_protocol.TestCryoCodexIntegration
```

To test actual inference with the backend installed and a GPU available:

```bash
CRYOCODEX_TEST_GPU=1 scipion3 test cryocodex.tests.test_protocol.TestCryoCodexGPU
```

## Citation

Cheng B, Su B, Yang J. *CryoCodex: learning discrete structural representations
for cryo-EM map post-processing*. bioRxiv (2026).
[doi:10.64898/2026.09.30.755675](https://doi.org/10.64898/2026.09.30.755675).

## License

The plugin is distributed under the MIT License. CryoCodex backend software and
model files are distributed separately; see the upstream project for their terms.
