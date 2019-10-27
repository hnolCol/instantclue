# -*- mode: python -*-

block_cipher = None


a = Analysis(['/Users/hnolte/Documents/GitHub/instantclue/src/instant_clue.py'],
             pathex=['/Users/hnolte/Documents/GitHub/instantclue/binaries/mac'],
             binaries=[('/System/Library/Frameworks/Tk.framework/Tk', 'tk'),
		                   ('/System/Library/Frameworks/Tcl.framework/Tcl', 'tcl')],
             datas=[],
             hiddenimports=['sklearn.utils.murmurhash', 'sklearn.neighbors.typedefs',
             				'sklearn.neighbors.quad_tree','sklearn.tree._utils',
             				'scipy._lib.messagestream','numpy.random.common',
                    'numpy.random.bounded_integers','numpy.random.entropy',
                    'sklearn.utils._cython_blas'],
             hookspath=[],#'/Users/hnolte/Documents/GitHub/instantclue/pyinstaller-hooks/'],
             runtime_hooks=[],#['pyinstaller-hooks/pyi_rth__tkinter.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Instant Clue',
          #icon = "/Users/hnolte/Documents/GitHub/instantclue/Logo.icns",
          debug=False,
          strip=False,
          clean=True,
          upx=True,
          runtime_tmpdir=None,
          console=True)
