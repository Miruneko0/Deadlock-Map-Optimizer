
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a44efcaa-5eb3-4c63-ad94-955411cff1b4" />



# Deadlock-Map-Optimizer
map optimization  deadlock

# Map Crop
<img width="1920" height="1080" alt="testrenderaddon2" src="https://github.com/user-attachments/assets/36c323ae-9592-4dc3-858e-0c6cef403d55" />

An add-on for Blender 5.1+ that simplifies the handling of large imported maps (such as Deadlock maps): it keeps only the desired portion of the map and what the camera sees.

Panel: 3D Viewport → Sidebar (N) → Map Crop.

## Features

**Crop by Zone**
- Add Zone Box: The zone is defined by one or more boxes.
- Crop to Zone: Everything outside the zone is hidden in the MAP_OFF collection; objects on the boundary are cropped precisely along the box edges, preserving UVs, colors, and normals.
- Boundary modes: precise cut, entire polygons, entire objects.
- The background (sky, clouds, water) remains intact.

**Crop by Camera**
- Crop to Camera: Only what fits within the frame remains, with a margin for angle and distance. Camera animation, constraints, and camera markers are taken into account.
- Current Frame: Precise crop based on the current frame.
- Hide Occluded: Hides objects obscured by other geometry. Transparent materials (glass, foliage) do not block the view.
- Preserve Lighting: Objects that are not visible but cast shadows, block the sky, reflect light, or emit light themselves remain invisible to the camera, and the lighting in the render remains unchanged.

**Render Acceleration**
- Cycles Culling: Enables culling of off-screen objects in Cycles.
- Persistent Data: The scene is not rebuilt for every frame of the animation.
- Tune Render: Automatically selects faster Cycles settings by comparing test renders to the original, ensuring the image remains unchanged.
- Fit Textures: scales textures down to the size actually needed on screen.



**Safety**
- Nothing is deleted until you confirm: “Restore Full Map” restores the scene to its original state.
- “Confirm (Delete Polygons)” permanently deletes hidden objects along with unused meshes, materials, and images.
- “Tune Render” and “Fit Textures” can be canceled using their respective buttons.

## Installation

Edit -> Preferences -> Add-ons -> Install from Disk → select `map_crop.py`.

## tests

Without addon with normal map render time: 1.5h in cycles and 30 min in eevee
With addon - zone box- 5 sec in eevee, cycles - 45 sec

By Miruneko

Discord - m1ru_neko

map by spenzo

Good discord server: https://discord.gg/G2BF3xpYZv



