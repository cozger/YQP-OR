# Unicode Symbols Guide

## Why Unicode Symbols Instead of Emoji?

This application uses **Unicode symbols** instead of **color emoji** for the following reasons:

1. **Cross-platform compatibility**: Works on Windows, Linux, macOS without special emoji fonts
2. **Smaller bundle size**: No need to include 10+ MB emoji fonts
3. **PyInstaller friendly**: Simple font bundling with DejaVu Sans (742 KB)
4. **Consistent rendering**: Symbols look the same across all platforms
5. **Faster rendering**: No complex emoji rendering pipeline

## Color Emoji vs Unicode Symbols

| Feature | Color Emoji 🔄 🌙 ☀️ | Unicode Symbols ⟳ ☾ ☀ |
|---------|---------------------|------------------------|
| **Font requirement** | Noto Color Emoji (~10 MB) | DejaVu Sans (742 KB) |
| **Default on Linux** | ❌ No | ✅ Yes |
| **Default on Windows** | ⚠️ Partial | ✅ Yes (Segoe UI Symbol) |
| **PyInstaller size** | +10 MB | +742 KB |
| **Rendering speed** | Slower (complex) | Fast (vector) |
| **Cross-platform** | ❌ Inconsistent | ✅ Consistent |

## Symbol Mapping

### Before (Color Emoji) → After (Unicode Symbols)

| Purpose | Emoji | Unicode | Codepoint | Font Support |
|---------|-------|---------|-----------|--------------|
| **Refresh button** | 🔄 | ⟳ | U+27F3 | DejaVu, Segoe UI |
| **Dark mode** | 🌙 | ☾ | U+263E | DejaVu, Segoe UI |
| **Light mode** | ☀️ | ☀ | U+2600 | DejaVu, Segoe UI |
| **Loading (1)** | ⏳ | ◷ | U+25F7 | DejaVu, Segoe UI |
| **Loading (2)** | ⌛ | ◶ | U+25F6 | DejaVu, Segoe UI |
| **Loading (3)** | - | ◵ | U+25F5 | DejaVu, Segoe UI |
| **Loading (4)** | - | ◴ | U+25F4 | DejaVu, Segoe UI |

## Unicode Character Details

### ⟳ - Clockwise Gapped Circle Arrow (U+27F3)
- **Block**: Supplemental Arrows-A
- **Category**: Symbol, Other
- **Fonts**: DejaVu Sans, Segoe UI Symbol, Arial Unicode MS
- **Usage**: Refresh/reload action

### ☾ - Last Quarter Moon (U+263E)
- **Block**: Miscellaneous Symbols
- **Category**: Symbol, Other
- **Fonts**: DejaVu Sans, Segoe UI Symbol, Arial Unicode MS
- **Usage**: Dark mode indicator

### ☀ - Black Sun with Rays (U+2600)
- **Block**: Miscellaneous Symbols
- **Category**: Symbol, Other
- **Fonts**: DejaVu Sans, Segoe UI Symbol, Arial Unicode MS
- **Usage**: Light mode indicator

### ◷ ◶ ◵ ◴ - Clock Face Quadrants (U+25F7-25F4)
- **Block**: Geometric Shapes
- **Category**: Symbol, Other
- **Fonts**: DejaVu Sans, Segoe UI Symbol
- **Usage**: Rotating loading spinner animation

## Font Fallback Strategy

The application uses font fallback to ensure symbols display correctly:

```python
# Loading spinner with font fallback
spinner_label = tk.Label(
    text="◷",
    font=('DejaVu Sans', 'Segoe UI Symbol', 36)
)
```

**Fallback order:**
1. **DejaVu Sans** (Linux, bundled) - Primary
2. **Segoe UI Symbol** (Windows) - Secondary
3. System default - Tertiary

## Testing Symbol Rendering

You can test symbol rendering in Python:

```python
import tkinter as tk

root = tk.Tk()
root.title("Symbol Test")

symbols = [
    ("Refresh", "⟳", "U+27F3"),
    ("Dark Mode", "☾", "U+263E"),
    ("Light Mode", "☀", "U+2600"),
    ("Loading 1", "◷", "U+25F7"),
    ("Loading 2", "◶", "U+25F6"),
]

for name, symbol, code in symbols:
    label = tk.Label(
        root,
        text=f"{symbol} {name} ({code})",
        font=('DejaVu Sans', 'Segoe UI Symbol', 16)
    )
    label.pack(pady=5)

root.mainloop()
```

## Bundled Font: DejaVu Sans

**Location**: `main/resources/fonts/DejaVuSans.ttf`

**Properties:**
- **Size**: 742 KB
- **Glyphs**: 3,310
- **Unicode coverage**: Excellent (Latin, Greek, Cyrillic, symbols)
- **License**: Free (Bitstream Vera License + Arev Fonts License)
- **Platform**: Cross-platform (Linux, Windows, macOS)

**Why DejaVu Sans?**
- Default on most Linux distributions
- Excellent Unicode symbol coverage
- Small file size compared to comprehensive Unicode fonts
- Widely used and well-tested
- Open source and free to redistribute

## Alternative Symbols (If Needed)

If symbols don't render correctly, consider these alternatives:

| Purpose | Alternative 1 | Alternative 2 | Alternative 3 |
|---------|---------------|---------------|---------------|
| Refresh | ↻ (U+21BB) | ⟲ (U+27F2) | ⇄ (U+21C4) |
| Dark | ◐ (U+25D0) | ● (U+25CF) | ○ (U+25CB) |
| Light | ◯ (U+25EF) | ☼ (U+263C) | ⊙ (U+2299) |
| Loading | ◴◵◶◷ | ●○○○ | ⬤⬤⬤⬤ |

## Implementation Notes

### Where Symbols Are Used

**gui.py locations:**
- Line 1020: Refresh button (`⟳ Refresh Cameras`)
- Line 1028: Theme toggle button (`☾ Dark Mode`)
- Line 1035: Theme toggle (dark) (`☀ Light Mode`)
- Line 2048: Toggle to dark (`☀ Light Mode`)
- Line 2053: Toggle to light (`☾ Dark Mode`)
- Line 2090: Loading spinner (`◷`)
- Line 2136: Spinner animation (`['◷', '◶', '◵', '◴']`)

### Font Configuration

Symbols use explicit font fallback in spinner (line 2091):
```python
font=('DejaVu Sans', 'Segoe UI Symbol', 36)
```

Buttons inherit font from Azure ttk theme (uses system default with fallback).

## Resources

- [Unicode Character Table](https://unicode-table.com/)
- [DejaVu Fonts](https://dejavu-fonts.github.io/)
- [Segoe UI Symbol (Windows)](https://docs.microsoft.com/en-us/typography/font-list/segoe-ui-symbol)
- [Unicode Geometric Shapes Block](https://unicode-table.com/en/blocks/geometric-shapes/)
- [Unicode Miscellaneous Symbols Block](https://unicode-table.com/en/blocks/miscellaneous-symbols/)
