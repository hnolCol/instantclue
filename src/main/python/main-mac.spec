
# -*- mode: python ; coding: utf-8 -*-
import shutil 
block_cipher = None


a = Analysis(['main.py'],
             pathex=['/Users/hnolte/Documents/GitHub/instantclue/src/main/python'],
             binaries=[],
             datas=[],
             hooksconfig={
                "matplotlib": {"backends": "all"}},
             hiddenimports=['setuptools','pynndescent','pkg_resources','sklearn.utils.murmurhash', 'sklearn.neighbors.typedefs','sklearn.neighbors._typedefs',
             				'sklearn.neighbors.quad_tree','sklearn.tree._utils',
             				'scipy._lib.messagestream','numpy.random.common',
                                   'numpy.random.bounded_integers','numpy.random.entropy','scipy.special.cython_special',
                                   'sklearn.utils._cython_blas','openTSNE._matrix_mul','openTSNE._matrix_mul.matrix_mul',
                                   'sklearn.metrics._pairwise_distances_reduction._datasets_pair','sklearn.metrics._pairwise_distances_reduction._middle_term_computer'],
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
             icon = '/Users/hnolte/Documents/GitHub/instantclue/src/main/python/icons/IC.icns', 
             version='0.12.0',
             bundle_identifier='com.nolte.instantclue',
             info_plist={
                'NSHighResolutionCapable': 'True',
                'NSPrincipalClass': 'NSApplication',
                'NSAppleScriptEnabled': False,
                'LSBackgroundOnly' : False,
                'UIUserInterfaceStyle': 'Light'
                }
            )


shutil.copyfile("/Users/hnolte/Documents/GitHub/instantclue/src/main/python/dist/stopwords","/Users/hnolte/Documents/GitHub/instantclue/src/main/python/dist/InstantClue.app/Contents/MacOS/wordcloud/stopwords")


