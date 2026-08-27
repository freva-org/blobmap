# blobmap logo

Two blobs of different sizes, each holding chunks. That is what blobmap does:
it decides which objects move together, and blobs are not all the same size.

| file | use | notes |
|---|---|---|
| `logo.svg` | mkdocs-material header | white, since the header sits on the primary colour |
| `logo-color.svg` | README, docs body, light backgrounds | full palette |
| `favicon.svg` | browser tab, small sizes | one blob, square, legible at 16px |

## Colours

    primary  #1f8fa5
    accent   #7bc6d6

These match `blobmap/theme.py`, which styles the CLI help. Change both
together, and replace them with the real waterpark palette -- these are a
guess at what fits a site called waterpark, not the actual site colours.

## Why three files

`currentColor` does not inherit through an `<img>`, which is how
mkdocs-material loads a logo, so a single adaptive file is not possible. The
header version is therefore explicitly white.

The favicon is a different drawing rather than the same one scaled. The two
blob mark is a tall, narrow composition that wastes a square viewport and
turns to mush below about 24px; the favicon keeps one blob and fills the
square.

The chunks are holes cut with `fill-rule="evenodd"`, not white rectangles, so
the mark works on any background.

## mkdocs

    theme:
      logo: assets/logo.svg
      favicon: assets/favicon.svg
