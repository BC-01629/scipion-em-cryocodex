# Changes

## 0.3.2

- Refresh the PyPI description with published installation, upgrade, configuration and verification instructions.
- Make README references usable on PyPI without repository-relative links.
- No protocol or coordinate-handling changes from 0.3.1.

## 0.3.1

- Temporarily disable origin-source selection, input origin normalization and explicit Scipion output-origin synchronization; retain correction code as comments.
- Preserve raw input origin/start fields and invalidate old crop confirmations.

## 0.3.0

- Fix wizard discovery circular imports by initializing the Scipion project GUI before importing FormWindow.

- Preserve input-file coordinates by default instead of Scipion's automatic centered origin; retain an explicit metadata-origin option.
- Resolve MRC origin/start fields without double counting and share the coordinate choice with crop review.
- Use snake_case output names for new runs while retaining access to completed legacy results.
- Simplify Summary and show the complete CryoCodex reference with a working HTTPS DOI.
- Download the official v1.0.0 release tag archive and record the tag in installation provenance.

## 0.2.0

- Add an automatically opened native crop editor for Fast mode, with three projections,
  draggable boundaries, coordinate entry, reset and explicit confirmation.
- Persist reviewed bounds and detect changes to input data, coordinates and preprocessing.
- Run the installed backend with confirmed bounds without modifying its files or waiting on stdin.
- Remove inference-window padding from small reviewed crops before extent restoration.
- Add crop regression tests and a reviewed-crop GPU test.
- Display the menu entry as `CryoCodex_enhance`.

## 0.1.0

- Initial Scipion plugin for CryoCodex 1.0.0.
- Pin backend source to a35e73a72f9cbc2b13cf6f8d95cf963cb704ca17.
- Install an isolated Python 3.11 runtime and model through Scipion installb.
- Expose Fast/Normal inference, batch size, stride, extent restoration and resampling.
- Register four outputs with their actual MRC grid coordinates and sampling.
- Add offline interface tests and an opt-in real GPU inference test.
