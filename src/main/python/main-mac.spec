
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['/Users/hnolte/Documents/GitHub/instantclue/src/main/python'],
             binaries=[],
             datas=[],
             hiddenimports=['pynndescent','pkg_resources','sklearn.utils.murmurhash', 'sklearn.neighbors.typedefs','sklearn.neighbors._typedefs',
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
a.datas += Tree('./annotations', prefix='annotations')
a.datas += Tree('./conf', prefix='conf')
a.datas += Tree('./icons', prefix='icons')
a.datas += Tree('./quickSelectLists', prefix='quickSelectLists')

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='InstantClue',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='InstantClue-mac')
app = BUNDLE(coll,
             name='InstantClue.app',
             icon=None,
             version='0.11.0',
             bundle_identifier='instantclue.de',
             info_plist={
                'NSHighResolutionCapable': 'True',
                'NSPrincipalClass': 'NSApplication',
                'NSAppleScriptEnabled': False,
                }
            )
