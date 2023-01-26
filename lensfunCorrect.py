#import lensfunpy

#cam_maker = 'NIKON CORPORATION'
#cam_model = 'NIKON D3S'
#lens_maker = 'Nikon'
#lens_model = 'Nikkor 28mm f/2.8D AF'

#db = lensfunpy.Database()
#cam = db.find_cameras(cam_maker, cam_model)[0]
#lens = db.find_lenses(cam, lens_maker, lens_model)[0]

#print(cam)
# Camera(Maker: NIKON CORPORATION; Model: NIKON D3S; Variant: ;
#        Mount: Nikon F AF; Crop Factor: 1.0; Score: 0)

#print(lens)
# Lens(Maker: Nikon; Model: Nikkor 28mm f/2.8D AF; Type: RECTILINEAR;
#      Focal: 28.0-28.0; Aperture: 2.79999995232-2.79999995232;
#      Crop factor: 1.0; Score: 110)

import cv2 
import lensfunpy

db = lensfunpy.Database()
cam = db.find_cameras('GoPro',"HERO4 Silver")[0]
print (cam)
#Camera(Maker: GoPro; Model: HERO4 Silver; Mount: goProHero4; Crop Factor: 7.659999847412109; Score: 0)

lens = db.find_lenses(cam)[0]
print(lens)
#Lens(Maker: GoPro; Model: HERO4; Type: FISHEYE_EQUISOLID; Focal: 3.0-3.0; Aperture: None-None; Crop factor: 7.659999847412109; Score: 15)


focal_length = 7.0
aperture = 0
distance = 50
image_path = 'testframe.png'
undistorted_image_path = 'out.png'

im = cv2.imread(image_path)
height, width = im.shape[0], im.shape[1]
print (height, width)
#1124 1998

mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)

mod.initialize(focal_length, aperture, distance, scale=0.85)

undist_coords = mod.apply_geometry_distortion()
im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
cv2.imwrite(undistorted_image_path, im_undistorted)

cv2.imshow('undistort', im_undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
