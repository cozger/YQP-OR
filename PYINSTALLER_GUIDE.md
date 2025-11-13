# PyInstaller Distribution Guide

This guide explains how to package YouQuantiPy as a standalone executable using PyInstaller.

## Prerequisites

```bash
pip install pyinstaller
```

## Font Bundling for Cross-Platform Compatibility

The application uses Unicode symbols (☾ ☀ ⟳ ◷) instead of color emoji for cross-platform compatibility. These symbols are supported by:

- **DejaVu Sans** (Linux) - Bundled in `resources/fonts/DejaVuSans.ttf`
- **Segoe UI Symbol** (Windows) - System font
- **Arial Unicode MS** (macOS) - System font

The bundled DejaVu Sans font (742KB) ensures consistent symbol rendering across all platforms.

## Building the Executable

### Method 1: Using PyInstaller Command Line

```bash
cd /home/canoz/Projects/youquantipy_mediapipe/main

pyinstaller \
  --onefile \
  --windowed \
  --name YouQuantiPy \
  --icon=resources/icon.ico \
  --add-data "themes:themes" \
  --add-data "resources/fonts:resources/fonts" \
  --add-data "models:models" \
  gui.py
```

**Flags explained:**
- `--onefile`: Create a single executable file
- `--windowed`: No console window (GUI app)
- `--add-data "source:destination"`: Bundle resources
  - Syntax: Windows uses `;`, Linux/Mac uses `:`

### Method 2: Using Spec File (Recommended)

Create `youquantipy.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('themes', 'themes'),                           # Azure theme files
        ('resources/fonts', 'resources/fonts'),          # DejaVu Sans font
        ('models', 'models'),                            # MediaPipe models
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YouQuantiPy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

Build with spec file:

```bash
pyinstaller youquantipy.spec
```

## Resource Path Resolution

The application uses `resource_path()` helper function (defined in `gui.py:146`) to handle resource paths correctly whether running from source or as a compiled executable:

```python
def resource_path(relative_path: str) -> str:
    """Get absolute path to resource for PyInstaller compatibility."""
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
```

**Example usage:**
```python
theme_file = resource_path(os.path.join('themes', 'azure.tcl'))
font_path = resource_path(os.path.join('resources', 'fonts', 'DejaVuSans.ttf'))
```

## Unicode Symbol Reference

The GUI uses these Unicode symbols (not color emoji):

| Symbol | Codepoint | Usage | Fallback Fonts |
|--------|-----------|-------|----------------|
| ⟳ | U+27F3 | Refresh button | DejaVu Sans, Segoe UI Symbol |
| ☾ | U+263E | Dark mode toggle | DejaVu Sans, Segoe UI Symbol |
| ☀ | U+2600 | Light mode toggle | DejaVu Sans, Segoe UI Symbol |
| ◷ | U+25F7 | Loading spinner | DejaVu Sans, Segoe UI Symbol |
| ◶ | U+25F6 | Loading spinner | DejaVu Sans, Segoe UI Symbol |
| ◵ | U+25F5 | Loading spinner | DejaVu Sans, Segoe UI Symbol |
| ◴ | U+25F4 | Loading spinner | DejaVu Sans, Segoe UI Symbol |

## Platform-Specific Notes

### Linux / WSL2
- DejaVu Sans bundled with application
- No additional fonts needed

### Windows
- Uses Segoe UI Symbol (system font) for symbols
- DejaVu Sans used as fallback if Segoe UI unavailable

### macOS
- Uses SF Symbols or Arial Unicode MS (system fonts)
- DejaVu Sans used as fallback

## Testing the Executable

After building:

```bash
# Linux
./dist/YouQuantiPy

# Windows
dist\YouQuantiPy.exe
```

## Troubleshooting

### Symbols not displaying
- Verify DejaVu Sans font is bundled: Check `dist/YouQuantiPy/_internal/resources/fonts/`
- Check console output for font loading errors
- Ensure `resource_path()` is used for all resource access

### Large executable size
- Remove unused dependencies in spec file (`excludes` list)
- Use UPX compression (`upx=True`)
- Consider using `--onedir` instead of `--onefile` for faster startup

### Import errors
- Add missing modules to `hiddenimports` in spec file
- Check PyInstaller hooks for complex dependencies (OpenCV, MediaPipe)

## Distribution Checklist

- [ ] Test on clean system without Python installed
- [ ] Verify all Unicode symbols render correctly
- [ ] Check theme switching (light/dark mode)
- [ ] Test camera enumeration and video capture
- [ ] Verify MediaPipe model loading
- [ ] Test configuration file persistence

## File Size Estimates

| Component | Size |
|-----------|------|
| Base executable | ~50-100 MB |
| DejaVu Sans font | 742 KB |
| Azure theme assets | ~500 KB |
| MediaPipe models | ~3-5 MB |
| **Total** | **~60-110 MB** |
