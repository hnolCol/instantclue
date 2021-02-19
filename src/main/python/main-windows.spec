# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\HNolte\\Downloads\\instantclue-QTInstantClue\\src\\main\\python'],
             binaries=[],
             datas=[],
             hiddenimports=['pynndescent','sklearn.utils.murmurhash', 'sklearn.neighbors.typedefs','sklearn.neighbors._typedefs',
             				'sklearn.neighbors.quad_tree','sklearn.tree._utils',
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

a.datas += Tree('./examples', prefix='examples')
a.datas += Tree('./conf', prefix='conf')
a.datas += Tree('./icons', prefix='icons')
a.datas += Tree('./quickSelectLists', prefix='quickSelectLists')

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
          console=True,
          icon = 'C:\\Users\\HNolte\\Downloads\\instantclue-QTInstantClue\\src\\main\\python\\icons\ICLogo.ico')
