# Resources

## Relation with clevr-dataset-gen

This document introduces the difference between the settings of CLEVR and ours.

These files are not changed:
- materials/Metal.blend (only rename),
- materials/Rubber.blend,
- shapes/SmoothCube_v2.blend,
- shapes/SmoothCylinder.blend,
- shapes/Sphere.blend.

A new material is added:
- materials/Glass.blend.

`base_scene.blend` is modified, and we:
- add two cameras around the plane,
- add four lines on the plane.

`properties.json` is modified, and we:
- add a glass material,
- add a medium size,
- add rules to define a new coordinate,
- add colors for generating segmentation.

We add `values.json` which includes all modifiable attributes and values.

## Acknowledgement

Thanks to the authors of [clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen).