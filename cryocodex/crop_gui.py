"""Native Scipion/Tk crop editor; no browser, network server or GPU required."""

import copy
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import tkinter as tk
import uuid

import numpy as np
from PIL import Image, ImageTk
import pyworkflow as pw
from pyworkflow.gui import Window
from pyworkflow.gui.widgets import Button, HotButton
from pyworkflow.utils import Icon

from . import Plugin
from .convert import prepare_input
from .crop_review import input_identity, record_is_current, runtime_script
from .scripts.crop_support import validate_bounds, write_json


class CropReviewWindow(Window):
    """Modal editor. A valid confirmation is the only normal closing action."""

    VIEWS = {'XY': ('x', 'y'), 'XZ': ('x', 'z'), 'YZ': ('y', 'z')}

    def __init__(self, form):
        super().__init__('CryoCodex — Check crop', form, minsize=(820, 560))
        self.form = form
        self.ready = False
        self.confirmed = False
        self.bounds = None
        self.views = {}
        self.events = queue.Queue()
        self.status = tk.StringVar(value='Preparing automatic crop boundaries. Please wait…')
        self.notice = tk.StringVar(value='Confirm and save settings to close this window.')
        self.background = pw.Config.SCIPION_BG_COLOR
        self.root.configure(bg=self.background)
        self.root.protocol('WM_DELETE_WINDOW', self.request_close)
        self.root.bind('<Escape>', self.request_close)
        self.root.transient(form.root)
        self._build()
        self.root.geometry('1120x650+80+60')
        self.root.deiconify()
        self.root.grab_set()
        self.root.lift()
        self.root.after(100, self._start_prepare)

    def _build(self):
        header = tk.Label(self.root, text='CryoCodex · Check crop', anchor='w',
                          bg=pw.Config.SCIPION_MAIN_COLOR, fg='white', font=self.fontBig,
                          padx=12, pady=8)
        header.pack(fill='x')
        tk.Label(self.root, text='Fast mode removes background. Drag a box edge to resize; '
                 'drag inside the box to move it.', bg=self.background,
                 font=self.font, anchor='w', padx=12, pady=8).pack(fill='x')
        frames = tk.Frame(self.root, bg=self.background)
        frames.pack(fill='both', expand=True, padx=10)
        frames.rowconfigure(0, weight=1)
        for i, (name, axes) in enumerate(self.VIEWS.items()):
            frames.columnconfigure(i, weight=1, uniform='views')
            panel = tk.LabelFrame(frames, text=name + ' projection', bg=self.background,
                                  font=self.fontBold)
            panel.grid(row=0, column=i, sticky='nsew', padx=3)
            canvas = tk.Canvas(panel, width=330, height=330, bg='#eeeeee', highlightthickness=0)
            canvas.pack(fill='both', expand=True)
            canvas.bind('<Configure>', lambda event, n=name: self.draw(n))
            canvas.bind('<ButtonPress-1>', lambda event, n=name: self.press(n, event))
            canvas.bind('<B1-Motion>', lambda event, n=name: self.drag(n, event))
            canvas.bind('<ButtonRelease-1>', lambda event, n=name: self.release(n))
            self.views[name] = dict(canvas=canvas, axes=axes, drag=None)
        controls = tk.Frame(self.root, bg=self.background)
        controls.pack(fill='x', padx=12, pady=8)
        self.entries = {}
        for i, axis in enumerate('xyz'):
            tk.Label(controls, text=axis.upper() + ' range:', font=self.font,
                     bg=self.background).grid(row=0, column=i*3, padx=(6, 2))
            for j in (0, 1):
                var = tk.StringVar()
                entry = tk.Entry(controls, textvariable=var, width=7, font=self.font)
                entry.grid(row=0, column=i*3+j+1, padx=2)
                entry.bind('<Return>', self.apply_entries)
                self.entries[axis, j] = var
        Button(controls, 'Apply coordinates', Icon.ACTION_REFRESH,
               command=self.apply_entries).grid(row=0, column=9, padx=8)
        self.threshold = tk.DoubleVar(value=0.0)
        tk.Label(controls, text='Display threshold (preview only)', bg=self.background,
                 font=self.font).grid(row=1, column=0, columnspan=3, sticky='w')
        tk.Scale(controls, from_=0, to=1, resolution=0.01, orient='horizontal',
                 variable=self.threshold, command=lambda value: self.redraw(),
                 bg=self.background, length=240).grid(row=1, column=3, columnspan=4, sticky='ew')
        tk.Label(self.root, textvariable=self.status, anchor='w', bg=self.background,
                 font=self.font, padx=12).pack(fill='x')
        tk.Label(self.root, textvariable=self.notice, anchor='w', bg=self.background,
                 fg=pw.Config.SCIPION_MAIN_COLOR, font=self.font, padx=12).pack(fill='x')
        footer = tk.Frame(self.root)
        footer.pack(fill='x', pady=(8, 0))
        Button(footer, 'Reset automatic crop', Icon.ACTION_REFRESH,
               command=self.reset).pack(side='left', padx=10, pady=12)
        self.confirm_button = HotButton(footer, 'Confirm and save settings', Icon.ACTION_SAVE,
                                        command=self.confirm, state=tk.DISABLED)
        self.confirm_button.configure(font=self.fontBold, padx=18, pady=8)
        self.confirm_button.pack(side='right', padx=12, pady=10)

    def _start_prepare(self):
        try:
            self.identity = input_identity(self.form.protocol)
            errors = Plugin.validateInstallation()
            if errors:
                raise ValueError('\n'.join(errors))
            volume = self.form.protocol.inputVolume.get().clone()
            volume.setFileName(self.identity['path'])
            self.previous = (json.loads(self.form.protocol.cropReview.get())
                             if record_is_current(self.form.protocol) else None)
        except Exception as exc:
            self._failed(str(exc))
            return

        def worker():
            try:
                with tempfile.TemporaryDirectory(prefix='cryocodex-crop-') as directory:
                    source = str(Path(directory) / 'input.mrc')
                    # prepare_input(volume, source,
                    #               use_header_origin=self.identity['origin_source'] == 0)
                    prepare_input(volume, source)
                    command = [Plugin.getPython(), '-u', runtime_script(), 'prepare',
                               '--backend-home', Plugin.getHome(), '--input', source,
                               '--output', directory]
                    # Bound numerical-library thread use while the desktop prepares a preview.
                    env = Plugin.getEnviron()
                    env['OMP_NUM_THREADS'] = str(self.form.protocol.numberOfThreads.get() or 4)
                    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    if result.returncode:
                        raise RuntimeError(result.stdout[-5000:])
                    record = json.loads((Path(directory) / 'preview.json').read_text())
                    with np.load(Path(directory) / 'projections.npz', allow_pickle=False) as archive:
                        projections = {name: archive[name.lower()].copy() for name in self.VIEWS}
                self.events.put(('ready', (record, projections)))
            except Exception as exc:
                self.events.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        try:
            kind, value = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        if kind == 'error':
            self._failed(value)
            return
        self.record, projections = value
        self.default_bounds = copy.deepcopy(self.record['bounds'])
        self.bounds = copy.deepcopy(self.previous['bounds'] if self.previous else self.default_bounds)
        for name, data in projections.items():
            self.views[name]['data'] = data
        self.ready = True
        self.confirm_button.configure(state=tk.NORMAL)
        self.status.set('Boundaries use the 1 Å working grid. Changes apply to all three views.')
        self.sync_entries()
        self.redraw()

    def _failed(self, message):
        # With no preview there is nothing valid to confirm. Report the failure,
        # disable review and return to the form instead of trapping the user.
        self.form.protocol.cropReview.set('')
        self.form.setVar('checkCrop', False)
        self.root.grab_release()
        self.root.destroy()
        self.form.showError('Could not prepare the crop preview:\n\n' + message)

    def request_close(self, event=None):
        self.notice.set('Please click “Confirm and save settings” before closing.')
        self.root.bell()
        return 'break'

    def sync_entries(self):
        for axis in 'xyz':
            for j in (0, 1):
                self.entries[axis, j].set(str(self.bounds[axis][j]))

    def apply_entries(self, event=None):
        if not self.ready:
            return False
        try:
            bounds = {a: [int(self.entries[a, j].get()) for j in (0, 1)] for a in 'xyz'}
            self.bounds = validate_bounds(bounds, self.record['shape'])
        except ValueError as exc:
            self.notice.set(str(exc))
            return False
        self.notice.set('Confirm and save settings to close this window.')
        self.redraw()
        return True

    def reset(self):
        if self.ready:
            self.bounds = copy.deepcopy(self.default_bounds)
            self.sync_entries()
            self.redraw()

    def redraw(self):
        if self.ready:
            for name in self.views:
                self.draw(name)

    def draw(self, name):
        if not self.ready:
            return
        view = self.views[name]
        canvas, data = view['canvas'], view['data']
        h, w = data.shape
        available_w, available_h = max(10, canvas.winfo_width()-24), max(10, canvas.winfo_height()-32)
        scale = min(available_w / w, available_h / h)
        width, height = max(1, round(w*scale)), max(1, round(h*scale))
        x0, y0 = (canvas.winfo_width()-width)/2, (canvas.winfo_height()-height)/2
        # Cache the density raster; dragging only redraws vector boundaries.
        key = (width, height, self.threshold.get())
        if view.get('raster_key') != key:
            high = max(float(data.max()), 1e-8)
            gray = np.sqrt(np.clip(data/high, 0, 1))
            gray[data < self.threshold.get()*high] = 0
            image = Image.fromarray(np.uint8(np.flipud(gray)*255))
            image = image.resize((width, height), Image.BILINEAR)
            view['photo'] = ImageTk.PhotoImage(image, master=self.root)
            view['raster_key'] = key
        canvas.delete('all')
        canvas.create_image(x0, y0, anchor='nw', image=view['photo'])
        u, v = view['axes']
        sx, sy = width/w, height/h
        left, right = x0+self.bounds[u][0]*sx, x0+(self.bounds[u][1]+1)*sx
        top, bottom = y0+(h-self.bounds[v][1]-1)*sy, y0+(h-self.bounds[v][0])*sy
        view['transform'] = x0, y0, sx, sy, h
        view['rectangle'] = left, top, right, bottom
        canvas.create_rectangle(left, top, right, bottom, outline='#00dddd', width=2)
        for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
            canvas.create_rectangle(x-3, y-3, x+3, y+3, fill='#00dddd', outline='white')
        canvas.create_text(x0+4, y0+height+4, anchor='nw',
                           text='%s →     %s ↑' % (u.upper(), v.upper()), font=self.font)

    def press(self, name, event):
        if not self.ready:
            return
        view = self.views[name]
        left, top, right, bottom = view['rectangle']
        if not left-9 <= event.x <= right+9 or not top-9 <= event.y <= bottom+9:
            return
        edges = []
        u, v = view['axes']
        for axis, index, distance in ((u, 0, abs(event.x-left)), (u, 1, abs(event.x-right)),
                                      (v, 1, abs(event.y-top)), (v, 0, abs(event.y-bottom))):
            if distance <= 9:
                edges.append((axis, index))
        view['drag'] = dict(x=event.x, y=event.y, original=copy.deepcopy(self.bounds), edges=edges)

    def drag(self, name, event):
        view = self.views[name]
        move = view['drag']
        if not move:
            return
        u, v = view['axes']
        _, _, sx, sy, _ = view['transform']
        delta = {u: round((event.x-move['x'])/sx), v: round((move['y']-event.y)/sy)}
        sizes = dict(zip('xyz', reversed(self.record['shape'])))
        for axis in (u, v):
            old = move['original'][axis]
            if not move['edges']:
                shift = max(-old[0], min(sizes[axis]-1-old[1], delta[axis]))
                self.bounds[axis] = [n+shift for n in old]
            else:
                for a, index in move['edges']:
                    if a == axis:
                        lo, hi = (0, old[1]) if index == 0 else (old[0], sizes[axis]-1)
                        self.bounds[axis][index] = max(lo, min(hi, old[index]+delta[axis]))
        self.sync_entries()
        self.redraw()

    def release(self, name):
        self.views[name]['drag'] = None

    def confirm(self):
        if not self.ready or not self.apply_entries():
            return
        try:
            if input_identity(self.form.protocol) != self.identity:
                raise ValueError('Input changed while reviewing. Restore the input or reopen the task.')
            record = dict(self.record, bounds=copy.deepcopy(self.bounds), input_identity=self.identity)
            directory = Path(self.form.protocol.getProject().path) / '.cryocodex-crops'
            directory.mkdir(exist_ok=True)
            # Save immediately even if the protocol form has not yet been saved.
            write_json(directory / (uuid.uuid4().hex + '.json'), record)
            self.form.protocol.cropReview.set(json.dumps(record, sort_keys=True))
        except (OSError, ValueError) as exc:
            self.notice.set('Could not save: ' + str(exc))
            return
        self.confirmed = True
        self.root.grab_release()
        self.root.destroy()
