# Development guide

## Release status

Version 0.3.1 is published on PyPI and has been verified after installation from
PyPI. Version 0.3.2 is a documentation update prepared for upload. Protocol behavior
is unchanged. The supported backend is CryoCodex v1.0.0; the tested environment is
Linux with Scipion 3.8.3.

Publishing to PyPI and inclusion in Scipion's official plugin catalog are separate steps.

## Repository layout

| Path | Purpose |
|---|---|
| `README.md` | User installation, usage and verification |
| `CHANGELOG.md` | Release history |
| `pyproject.toml` | Package metadata, dependencies and Scipion entry point |
| `MANIFEST.in` | Source distribution contents |
| `cryocodex/__init__.py` | Plugin registration, configuration and backend installation |
| `cryocodex/constants.py` | Backend version, download URLs and output names |
| `cryocodex/convert.py` | Input conversion and map geometry |
| `cryocodex/crop_gui.py` | Interactive crop window |
| `cryocodex/crop_review.py` | Crop records and input identity checks |
| `cryocodex/wizards.py` | Wizard registration and automatic window callback |
| `cryocodex/protocols/` | Processing protocol |
| `cryocodex/scripts/` | Backend installer and crop adapters |
| `cryocodex/tests/` | Unit, integration and GPU tests, including a synthetic backend |

Build artifacts, Python caches, local environments and `.local/` are excluded from
Git. Earlier development notes are retained under `.local/` for local reference.
Do not add backend environments, model checkpoints, input maps or Scipion projects
to the source repository.

## Editable installation

```bash
scipion3 installp -p /absolute/path/to/scipion-em-cryocodex --devel --noBin
```

Restart Scipion after source changes. Reinstall the editable package when changing
its dependencies, version or entry-point metadata. Install the backend separately:

```bash
scipion3 installb cryocodex-1.0.0
```

The backend installer creates a separate Python 3.11 environment and downloads the
model. See README for configuration of an existing installation.

## Retained coordinate correction code

Since 0.3.1, the plugin does not undo coordinate changes made during Scipion import.
Prepared inputs retain the file origin and axis-mapped start indices. Outputs do
not explicitly set the Scipion Volume origin. Cropping, resampling, sampling-rate
handling and output-grid consistency checks remain enabled.

The previous correction code is retained as comments in:

- `protocols/protocol_enhance.py`: origin-source parameter, input preparation call
  and output `setOrigin` calls.
- `convert.py`: previous function signature, origin selection and start normalization.
- `crop_gui.py` and `crop_review.py`: crop-preview calls and input identity fields.

To restore correction, update these parts together, restore the `originSource`
change listener in `wizards.py`, update tests and increment the crop record's
`coordinate_version` (currently 3). Existing crop confirmations must be reviewed
again. Restoring only output `setOrigin` does not restore the entire workflow.

Wizard initialization imports `pyworkflow.gui.project` before `FormWindow` to avoid
a circular import during inspection in Scipion 3.8.3. Regression tests cover both
plugin-first and project-GUI-first import orders.

## Tests

Checks that do not require a GPU:

```bash
scipion3 inspect cryocodex
scipion3 python -m unittest cryocodex.tests.test_unit cryocodex.tests.test_crop cryocodex.tests.test_wizards
scipion3 test cryocodex.tests.test_protocol.TestCryoCodexIntegration
```

The integration test uses a synthetic backend. To test real inference, use a GPU
session with the backend and model installed:

```bash
CRYOCODEX_TEST_GPU=1 scipion3 test cryocodex.tests.test_protocol.TestCryoCodexGPU
```

Manually check the Fast-mode crop window, confirmation and reopening in the GUI.
With a GUI-confirmed crop record, run:

```bash
CRYOCODEX_TEST_REVIEW=/absolute/path/to/crop_review.json \
scipion3 test cryocodex.tests.test_protocol.TestCryoCodexReviewedGPU
```

Version 0.3.1 passed 22 unit/regression tests, the synthetic-backend integration
test and real GPU inference. This does not establish coverage of all maps or
parameter combinations. Documentation-only changes do not require repeating GPU
tests; protocol, crop or coordinate changes require relevant validation.

## GitHub setup

Create an empty GitHub repository named `scipion-em-cryocodex`, without generating
a README or license. The local repository uses the `main` branch.

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Prepare CryoCodex Scipion plugin repository"
git remote add origin https://github.com/YOUR_ACCOUNT/scipion-em-cryocodex.git
git push -u origin main
```

Replace `YOUR_ACCOUNT` with your actual user or organization. Configure your Git
author name and email if requested. After creating the repository, add its actual
URL and issue tracker to `[project.urls]` in `pyproject.toml`. The existing
`CryoCodex source` URL refers to the backend, not this plugin.

## Releases

1. Update the version in both `pyproject.toml` and `cryocodex/__init__.py`.
2. Update CHANGELOG and README, then run the relevant tests.
3. Build and inspect the distributions in a separate tools environment.

```bash
python3 -m venv .venv-release
.venv-release/bin/python -m pip install --upgrade build twine
.venv-release/bin/python -m build
```

Replace `X.Y.Z` below with the new version. Select only that version's artifacts:

```bash
.venv-release/bin/python -m twine check --strict dist/scipion_em_cryocodex-X.Y.Z*
.venv-release/bin/python -m twine upload -c "scipion-3.8.3" dist/scipion_em_cryocodex-X.Y.Z*
```

Verify installation from the new wheel before uploading. Enter the PyPI API token
at the terminal prompt; do not store it in the repository. After publication,
reinstall from PyPI and verify, then tag the corresponding source commit.

PyPI displays the README embedded in the uploaded distribution. Editing the GitHub
README does not update PyPI. Publish a new version to update that description;
do not overwrite existing release files. Keep published artifacts unchanged.
