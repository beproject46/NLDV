import os
os.system('ffmpeg -i /home/foo/Project/input/clab.mp4 -vf "select=gt(scene\,0.3)" -vsync vfr /home/foo/Project/Dataset/%02d.jpg')
