"""Scipion wizard registration and a protocol-scoped automatic form callback."""

from functools import wraps

from pyworkflow.wizard import Wizard

from .protocols import ProtCryoCodexEnhance


def open_crop(form):
    from .crop_gui import CropReviewWindow
    existing = getattr(form, '_cryocodex_crop_window', None)
    if existing is not None and existing.root.winfo_exists():
        existing.root.lift()
        return
    if form.protocol.mode.get() != 0:
        return
    form._cryocodex_crop_window = CropReviewWindow(form)


class CryoCodexCropWizard(Wizard):
    _targets = [(ProtCryoCodexEnhance, ['checkCrop'])]

    def show(self, form, *params):
        # The usual wizard icon also lets the user reopen a confirmed crop.
        form.setVar('checkCrop', True)
        open_crop(form)


def _install_auto_open():
    """Scipion 3 has wizard buttons, but no plugin callback for a checkbox.

    Retain the normal form callback and add an idle callback ONLY for this
    protocol's three relevant fields. Never edit Scipion installation files.
    This small compatibility hook is exercised by the GUI regression test.
    """
    # Initialize the project package first: form imports project.utils,
    # whose package initializer imports viewprotocols and FormWindow again.
    # This is the same import order used by the Scipion project GUI.
    import pyworkflow.gui.project  # noqa: F401
    from pyworkflow.gui.form import FormWindow
    if getattr(FormWindow._checkChanges, '_cryocodex_auto_open', False):
        return
    original = FormWindow._checkChanges

    @wraps(original)
    def changed(form, name):
        result = original(form, name)
        if (isinstance(form.protocol, ProtCryoCodexEnhance)
                and name in ('checkCrop', 'mode', 'inputVolume')):
            form._cryocodex_review_reason = name
            if not getattr(form, '_cryocodex_review_queued', False):
                form._cryocodex_review_queued = True

                def on_idle():
                    form._cryocodex_review_queued = False
                    if not form.root.winfo_exists():
                        return
                    prot = form.protocol
                    if prot.isActive() or prot.mode.get() != 0 or not prot.checkCrop.get():
                        return
                    from .crop_review import record_is_current
                    if (form._cryocodex_review_reason == 'checkCrop'
                            or not record_is_current(prot)):
                        open_crop(form)

                form.root.after_idle(on_idle)
        return result

    changed._cryocodex_auto_open = True
    FormWindow._checkChanges = changed


_install_auto_open()
