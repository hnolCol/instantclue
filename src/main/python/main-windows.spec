# -*- mode: python ; coding: utf-8 -*-
import os, shutil
block_cipher = None

shutil.copytree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\examples',
'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\dist\\examples')

shutil.copytree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\conf',
'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\dist\\conf')

shutil.copytree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\annotations',
'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\dist\\annotations')

shutil.copytree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\icons',
'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\dist\\icons')

shutil.copytree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\quickSelectLists',
'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\dist\\quickSelectLists')


a = Analysis(['main.py'],
             pathex=['C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python'],
             binaries=[],
             datas=[],
             hiddenimports=['pynndescent','sklearn.utils.murmurhash', 'sklearn.neighbors.typedefs','sklearn.neighbors._typedefs',
             				'sklearn.neighbors.quad_tree','sklearn.tree._utils','sklearn.neighbors._partition_nodes',
             				'scipy._lib.messagestream','numpy.random.common',
                                   'numpy.random.bounded_integers','numpy.random.entropy','scipy.special.cython_special',
                                   'sklearn.utils._cython_blas','openTSNE._matrix_mul','openTSNE._matrix_mul.matrix_mul'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)





a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\examples', prefix='examples')
a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\annotations', prefix='annotations')
a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\conf', prefix='conf')
a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\conf\\key', prefix='key')
a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\icons', prefix='icons')
a.datas += Tree('C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\quickSelectLists', prefix='quickSelectLists')

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='InstantClue',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon = 'C:\\Users\\HNolte\\Documents\\GitHub\\instantclue\\src\\main\\python\\icons\ICLogo.ico')
