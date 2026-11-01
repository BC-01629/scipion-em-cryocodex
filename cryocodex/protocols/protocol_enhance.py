"""Run the pinned CryoCodex backend as a Scipion GPU protocol."""

import os
import shlex

import numpy as np
from pyworkflow.constants import BETA
from pyworkflow.protocol import params
from pwem.objects import Volume, VolumeMask
# from pwem.objects import Transform  # Restore with output origin synchronization.
from pwem.protocols import ProtAnalysis3D

from .. import Plugin
from ..constants import OUTPUT_FILES, CHIMERAX_COLOR_COMMAND
from ..convert import prepare_input, read_geometry


class ProtCryoCodexEnhance(ProtAnalysis3D):
    """Enhance a density map and predict local quality and a molecular mask.

    Fast mode crops the background. Sampling is read from each generated MRC.
    Explicit Scipion output-origin synchronization is temporarily disabled.
    """
    _label = "CryoCodex-Enhance"
    _devStatus = BETA
    _possibleOutputs = {
        "enhanced_volume": Volume, "input_volume_local_quality": Volume,
        "enhanced_volume_local_quality": Volume, "molecular_mask": VolumeMask,
    }

    @classmethod
    def getClassLabel(cls, prependPackageName=True):
        # The product label already includes CryoCodex; avoid a repeated prefix.
        return cls._label

    def _defineParams(self, form):
        form.addSection(label="Input")
        # Having GPU_LIST without USE_GPU declares a GPU-only protocol in Scipion.
        form.addHidden(params.GPU_LIST, params.StringParam, default="0",
                       label="GPU ID", help="One GPU per run; managed by Scipion's executor.")
        form.addParam("inputVolume", params.PointerParam, pointerClass="Volume",
                      important=True, label="Input volume")
        # Origin correction is temporarily disabled; retained for re-enabling.
        # form.addParam("originSource", params.EnumParam,
                      # choices=["Input file header", "Scipion volume metadata"], default=0,
                      # label="Origin source", expertLevel=params.LEVEL_ADVANCED,
                      # help="Preserve the input file coordinates by default: use MRC origin, "
                           # "or start indices if origin is zero. Scipion import may automatically "
                           # "center volumes. Select Scipion metadata only to use that coordinate frame.")
        form.addParam("mode", params.EnumParam, choices=["Fast (crop background)", "Normal (whole map)"],
                      default=0, label="Processing mode")
        form.addParam("checkCrop", params.BooleanParam, default=False, condition="mode==0",
                      label="Check crop (interactive)",
                      help="Fast mode removes background. Choose Yes to open the crop editor. "
                           "Drag the automatic boundaries, then confirm and save. "
                           "Use the wizard icon to review them again before Execute.")
        form.addHidden("cropReview", params.StringParam, default="")
        form.addParam("batchSize", params.IntParam, default=9,
                      label="Batch size", help="Reduce this value if GPU memory is insufficient.")
        form.addParam("stride", params.IntParam, default=12, label="Sliding-window stride",
                      help="1–48 voxels. Smaller strides process more overlapping boxes.")
        form.addParam("keepSize", params.BooleanParam, default=False, condition="mode==0",
                      label="Restore full map extent",
                      help="Fill cropped background with zeros on the working grid.")
        form.addParam("reverseInterpolation", params.BooleanParam, default=False,
                      label="Return to input voxel size",
                      help="Otherwise outputs use CryoCodex's 1 Å grid. Output dimensions may change.")
        form.addParallelSection(threads=4, mpi=1)

    def _insertAllSteps(self):
        self._insertFunctionStep(self.prepareInputStep, needsGPU=False)
        self._insertFunctionStep(self.runCryoCodexStep, needsGPU=True)
        self._insertFunctionStep(self.createOutputStep, needsGPU=False)

    def prepareInputStep(self):
        # prepare_input(self.inputVolume.get(), self._getTmpPath("input.mrc"),
        #               use_header_origin=self.originSource.get() == 0)
        prepare_input(self.inputVolume.get(), self._getTmpPath("input.mrc"))
        if self._usesReviewedCrop():
            from ..crop_review import confirmed_record
            from ..scripts.crop_support import map_signature, write_json
            record = confirmed_record(self)
            if map_signature(self._getTmpPath("input.mrc")) != record['input_sha256']:
                raise ValueError("Input data changed since crop confirmation. Check the crop again.")
            write_json(self._getExtraPath("crop_review.json"), record)

    def _usesReviewedCrop(self):
        return self.mode.get() == 0 and self.checkCrop.get()

    def _arguments(self):
        args = ["-i", os.path.abspath(self._getTmpPath("input.mrc")),
                "-o", os.path.abspath(self._getExtraPath()), "-n", "cryocodex",
                "-m", Plugin.getModel(), "-b", str(self.batchSize.get()),
                "-s", str(self.stride.get()), "--normal", str(self.mode.get() == 1),
                "--keep_size", str(self.mode.get() == 0 and self.keepSize.get()),
                "--reverse_interpolation", str(self.reverseInterpolation.get()),
                "--crop_check", "False", "--no_logo", "True"]
        # Scipion interpolates %(GPU)s at execution time. Preserve literal '%' in paths.
        if self._usesReviewedCrop():
            args = ["run", "--backend-home", os.path.abspath(Plugin.getHome()),
                    "--review-file", os.path.abspath(self._getExtraPath("crop_review.json")), "--"] + args
        return " ".join(shlex.quote(x) for x in args).replace("%", "%%") + " -g '%(GPU)s'"

    def runCryoCodexStep(self):
        env = Plugin.getEnviron()
        env["OMP_NUM_THREADS"] = str(self.numberOfThreads.get())
        program = Plugin.getProgram()
        if self._usesReviewedCrop():
            from ..crop_review import runtime_script
            program = shlex.quote(Plugin.getPython()) + " -u " + shlex.quote(runtime_script())
        self.runJob(program, self._arguments(), env=env,
                    cwd=self._getExtraPath(), numberOfMpi=1)

    def createOutputStep(self):
        outputs = {}
        reference = None
        for name, filename in OUTPUT_FILES.items():
            path = self._getExtraPath(filename)
            if not os.path.isfile(path):
                raise RuntimeError("CryoCodex did not produce required output: " + path)
            geometry = read_geometry(path)
            if reference is None:
                reference = geometry
            elif (geometry["dimensions"] != reference["dimensions"] or
                  not np.isclose(geometry["sampling"], reference["sampling"]) or
                  not np.allclose(geometry["origin"], reference["origin"], atol=1e-4)):
                raise RuntimeError("CryoCodex output maps have inconsistent grids: " + path)
            output = self._possibleOutputs[name]()
            output.setFileName(path)
            output.setSamplingRate(geometry["sampling"])
            # Disabled: let Scipion use its default output-origin behavior.
            # origin = Transform()
            # origin.setShifts(*geometry["origin"])
            # output.setOrigin(origin)
            outputs[name] = output
        self._defineOutputs(**outputs)
        for output in outputs.values():
            self._defineSourceRelation(self.inputVolume, output)
        for message in self._previewSummary():
            self.info(message)

    def _validate(self):
        errors = []
        if self.mode.get() not in (0, 1):
            errors.append("Unknown processing mode.")
        if self.batchSize.get() is None or self.batchSize.get() < 1:
            errors.append("Batch size must be a positive integer.")
        if self.stride.get() is None or not 1 <= self.stride.get() <= 48:
            errors.append("Stride must be between 1 and 48 voxels.")
        try:
            if len(self.getGpuList()) != 1:
                errors.append("CryoCodex supports exactly one GPU per run.")
        except (ValueError, TypeError):
            errors.append("Choose a single valid GPU ID.")
        volume = self.inputVolume.get()
        if volume is not None:
            sampling = volume.getSamplingRate()
            if sampling is None or not np.isfinite(sampling) or sampling <= 0:
                errors.append("Input sampling rate must be finite and positive.")
        if self._usesReviewedCrop():
            from ..crop_review import confirmed_record
            try:
                confirmed_record(self)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append("Confirm the crop in the wizard before execution: " + str(exc))
        return errors

    def _summary(self):
        if not self._hasOutputs():
            return ["CryoCodex outputs are not ready."]
        return self._previewSummary()

    def _hasOutputs(self):
        # Completed projects keep their persisted output names and connections.
        return hasattr(self, "enhanced_volume") or hasattr(self, "enhancedVolume")

    def _previewSummary(self):
        return [
            "Results saved in: " + os.path.abspath(self._getExtraPath()),
            "Local quality scores range from 0 to 1; higher scores indicate higher predicted local map quality.",
            "In ChimeraX, display the density maps as surfaces, then color the original map "
            "with inscore (cryocodex_in_score.mrc) and the enhanced map (cryocodex.mrc) "
            "with outscore (cryocodex_out_score.mrc) to inspect the network's predicted local quality.",
            "In the command below, #1 is the density map to color and #2 is its corresponding "
            "score map (inscore for the original map; outscore for the enhanced map). "
            "Replace #1 and #2 with the actual model IDs in ChimeraX.",
            CHIMERAX_COLOR_COMMAND,
        ]

    def _methods(self):
        return ["CryoCodex enhanced the input map and estimated local quality and a molecular mask "
                "[Cheng2026CryoCodex].",
                "Bin Cheng, Baoquan Su and Jianyi Yang. CryoCodex: learning discrete structural "
                "representations for cryo-EM map post-processing. bioRxiv (2026). "
                "[[https://doi.org/10.64898/2026.09.30.755675][doi:10.64898/2026.09.30.755675]]"] if self._hasOutputs() else []

    def _citations(self):
        return ["Cheng2026CryoCodex"]
