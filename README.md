# Demeter project website

Project page: https://tianhang-cheng.github.io/Demeter/

This standalone `gh-pages` branch contains the project website and its assets,
migrated from `Tianhang-Cheng.github.io/Demeter`. The research code remains on `main`.

In GitHub Settings > Pages, select **Deploy from a branch**, then **gh-pages**
and **/ (root)**. The `.nojekyll` file serves the HTML and assets directly.

To preview locally, run `python -m http.server 8000` in this directory and open
http://localhost:8000/. Keep `static/` alongside `index.html`; asset paths are relative.

## Asset optimization

The migrated assets were reduced from approximately 587 MB to 375 MB:
JPEG images use WebP quality 80 at their original dimensions; PNG diagrams use
lossless WebP. PLY models retain every vertex, face, normal and color, with double
fields converted to the same float32 precision used by Three.js for rendering.
The PDF and videos retain their original bytes.

`scripts/asset-report.json` records original and optimized file sizes.
To rebuild from a separate original website copy, install Pillow and NumPy and run
`python scripts/optimize_assets.py /path/to/original/site` from this directory.
The script replaces image assets and updates HTML/JavaScript references.

## Template attribution

This is the repository that contains source code for the [Nerfies website](https://nerfies.github.io).

If you find Nerfies useful for your work please cite:
```
@article{park2021nerfies
  author    = {Park, Keunhong and Sinha, Utkarsh and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Seitz, Steven M. and Martin-Brualla, Ricardo},
  title     = {Nerfies: Deformable Neural Radiance Fields},
  journal   = {ICCV},
  year      = {2021},
}
```

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
