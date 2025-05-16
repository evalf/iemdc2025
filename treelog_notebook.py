from contextlib import contextmanager
import IPython.display
import os.path
from pathlib import Path
from tempfile import NamedTemporaryFile
import treelog

class IPythonImageLog:
    'Output images via ``IPython.display.Image``.'

    def __init__(self):
        self.currentcontext = []

    def pushcontext(self, title):
        self.currentcontext.append(title)
        self.contextchangedhook()

    def popcontext(self):
        self.currentcontext.pop()
        self.contextchangedhook()

    def recontext(self, title):
        self.currentcontext[-1] = title
        self.contextchangedhook()

    def contextchangedhook(self):
        pass

    def write(self, text, level):
        print(' > '.join((*self.currentcontext, text)))

    @contextmanager
    def open(self, filename, mode, level):
        with NamedTemporaryFile('wb', suffix=os.path.splitext(filename)[1], delete_on_close=False) as f:
            yield f
            f.close()
            data = Path(f.name).read_bytes()
            if filename.endswith('.svg'):
                im = IPython.display.SVG(data)
            else:
                im = IPython.display.Image(data, embed=True)
            IPython.display.display(im)


_ctx = treelog.set(treelog.FilterLog(IPythonImageLog(), minlevel=treelog.proto.Level.info))
_ctx.__enter__()
