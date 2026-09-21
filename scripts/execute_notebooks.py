"""Execute the plain-Python notebooks in-process, without Jupyter kernel sockets.

This repository's cells contain ordinary Python (no magics or asynchronous cells).
IPython captures actual display/stream outputs; errors stop execution immediately.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root=Path(__file__).resolve().parents[1]
os.chdir(root)
for path in sorted((root/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    InteractiveShell.clear_instance()
    shell=InteractiveShell.instance();count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path.name}, cell {count}: {result.error_before_exec or result.error_in_exec}')
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.outputs=outputs;cell.execution_count=count
    nb.metadata['execution_method']='In-process IPython, actual captured cell outputs; no kernel sockets required.'
    nbformat.validate(nb);nbformat.write(nb,path)
    print('Executed',path.name,flush=True)
