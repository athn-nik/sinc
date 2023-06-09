import bpy

class Camera:
    def __init__(self, *, first_root, mode, is_mesh, fakeinone=False):
        camera = bpy.data.objects['Camera']

        ## initial position
        camera.location.x = 7.36
        camera.location.y = -6.93 
        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.8
        else:
            camera.location.z = 5.2

        # wider point of view
        if mode == "sequence":
            if is_mesh:
                if fakeinone:
                    camera.data.lens = 75
                else:
                    camera.data.lens = 100

            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 140
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 90
            else:
                camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root
    
    def eye_view(self):
    
        from math import radians
        cam_rot = bpy.data.objects["Camera"].rotation_euler

        self.camera.location.z += 0.7
        self.camera.location.y += 1.2
        self.camera.location.x -= 1.25
        self.camera.rotation_euler = (cam_rot[0] + radians(-10), cam_rot[1] + radians(0), cam_rot[2] + radians(0))

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
