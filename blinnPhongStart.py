from typing import Any
import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import ctypes
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize


############################## Constants ######################################

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_EXIT = 1

#0: debug, 1: production
GAME_MODE = 0

############################## helper functions ###############################

def initialize_glfw():

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    #for uncapped framerate
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER,GL_FALSE) 
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Title", None, None)
    glfw.make_context_current(window)
    
    #glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    glEnable(GL_PROGRAM_POINT_SIZE)
    glClearColor(0.1, 0.1, 0.1, 1)

    return window

###############################################################################

class Pyramid:


    def __init__(self, position, eulers, theta = 0, phi = 0):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        
        self.theta = theta
        self.phi = phi
        self.update_vectors()

    def get_pyramid_model_matrix(self):
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_eulers(
                eulers=np.radians(self.eulers), dtype=np.float32
            )
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.position),dtype=np.float32
            )
        )
        return model_transform
    
    def update_vectors(self):
        #self.forwards = np.array([0,0,0], dtype=np.float32)
        
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))
            ],
            dtype = np.float32
        )

        
        globalUp = np.array([0,0,1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)
    

class Light:
    def __init__(self, position, color, strength):

        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

class Player:


    def __init__(self, position):

        self.position = np.array(position, dtype = np.float32)
        self.theta = 0
        self.phi = 0
        #self.forwards = np.array([0,0,0], dtype=np.float32)
        self.update_vectors()
    
    def update_vectors(self):
        #self.forwards = np.array([0,0,0], dtype=np.float32)
        
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))
            ],
            dtype = np.float32
        )

        
        
        globalUp = np.array([0,0,1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)

class Scene:


    def __init__(self):

        self.pyramids = [
            Pyramid(
                position = [0,0,0],
                eulers = [0,0,0],
                theta=90,
                phi=0
            ),
        ]

        self.player = Player(
            position = [0,-5,0]
        )
        
        #rotate to face the triangle
        # THETA = Angle off the X axis, going left
        # PHI angle UP off the Y axis
        self.spin_player(90,0)
        
        self.lights = [
            Light(
                position = [
                    -1,
                    -2,
                    1
                ],
                color = [
                    0,
                    1,
                    0,
                    
                ],
                strength = 2
            ),
            
            Light(
                position = [
                    1,
                    1,
                    1
                ],
                color = [
                    1,
                    0,
                    0,
                    
                ],
                strength = 2
            ),
            
            Light(
                position = [
                    np.random.uniform(low=-5,high=5),
                    np.random.uniform(low=-1,high=5),
                    np.random.uniform(low=-2.0,high=4.0)
                ],
                color = [
                    np.random.uniform(low=0,high=1),
                    np.random.uniform(low=0,high=1),
                    0,
                    
                ],
                strength = 4
            ),
            Light(
                position = [
                    np.random.uniform(low=-5,high=5),
                    np.random.uniform(low=-1,high=5),
                    np.random.uniform(low=-2.0,high=4.0)
                ],
                color = [
                    np.random.uniform(low=0,high=1),
                    np.random.uniform(low=0,high=1),
                    0,
                    
                ],
                strength = 4
            ),
        ]

    def update(self, rate):
        '''
        
        for pyramid in self.pyramids:
            pyramid.eulers[2] += 0.25 * rate
            if pyramid.eulers[2] > 360:
                pyramid.eulers[2] -= 360
        '''
        
        

    def move_player(self, dPos):

        dPos = np.array(dPos, dtype = np.float32)
        self.player.position += dPos

    def spin_player(self, dTheta, dPhi):

        self.player.theta += dTheta
        if self.player.theta > 360:
            self.player.theta -= 360
        elif self.player.theta < 0:
            self.player.theta += 360
        
        self.player.phi = min(
            89, max(-89, self.player.phi + dPhi)
        )
        self.player.update_vectors()
    
    def move_pyramid(self, dPos):
        dPos = np.array(dPos, dtype = np.float32)
        self.pyramids[0].position += dPos

    def spin_pyramid(self, dTheta, dPhi):
        '''
        to spin the pyramid, we need to update the THETA which is used for the 
        forwards vector calculation. this makes sure when we move forward, our forwards
        vector is actually changing
        
        ALSO
        
        we need to rotate the pyramid on its eulers to change its orientation
        on the screen
        
        We also 
        
        '''
        self.pyramids[0].eulers[1] += dTheta
        if self.pyramids[0].eulers[1] > 360:
            self.pyramids[0].eulers[1] -= 360
        elif self.pyramids[0].eulers[1] < 0:
            self.pyramids[0].eulers[1] += 360
        
        reverseTheta = -1 * dTheta
        
        self.pyramids[0].theta += reverseTheta
        if self.pyramids[0].theta > 360:
            self.pyramids[0].theta -= 360
        elif self.pyramids[0].theta < 0:
            self.pyramids[0].theta += 360
        
        self.pyramids[0].phi = min(
            89, max(-89, self.pyramids[0].phi + dPhi)
        )
        self.pyramids[0].update_vectors()


    def rolling_arrow(self, rate):
        for pyramid in self.pyramids:
            pyramid.eulers[2] += 0.25 * rate
            if pyramid.eulers[2] > 360:
                pyramid.eulers[2] -= 360

    

class App:


    def __init__(self, window):

        self.window = window

        self.renderer = GraphicsEngine()

        self.scene = Scene()
        
        

        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.mainLoop()

    def mainLoop(self):
        running = True
        while (running):
            #check events
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False
            
            self.handleKeys()
            self.handleMouse()
            #print(str(self.scene.player.position))

            glfw.poll_events()

            self.scene.update(self.frameTime / 16.67)
            
            self.renderer.render(self.scene)

            #timing
            self.calculateFramerate()
        self.quit()

    def handleKeys(self):

        combo = 0
        directionModifier = 0
        """
        w: 1 -> 0 degrees
        a: 2 -> 90 degrees
        w & a: 3 -> 45 degrees
        s: 4 -> 180 degrees
        w & s: 5 -> x
        a & s: 6 -> 135 degrees
        w & a & s: 7 -> 90 degrees
        d: 8 -> 270 degrees
        w & d: 9 -> 315 degrees
        a & d: 10 -> x
        w & a & d: 11 -> 0 degrees
        s & d: 12 -> 225 degrees
        w & s & d: 13 -> 270 degrees
        a & s & d: 14 -> 180 degrees
        w & a & s & d: 15 -> x
        """

        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 8
        
        #USE SHIFT AND SPACE TO GO DOWN AND UP IN Z AXIS, RESPECTIVELy
        #represents the change in the cameras vertical positon (z) axis
        camera_z = 0
        
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_LEFT_SHIFT) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_z += -1
        
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_SPACE) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_z += 1
        
        dPos = [
                0,
                0,
                camera_z * 0.025 * 0.2
        ]
        self.scene.move_player(dPos)

        
        if (combo > 0):
            if combo == 3:
                directionModifier = 45
            elif combo == 2 or combo == 7:
                directionModifier = 90
            elif combo == 6:
                directionModifier = 135
            elif combo == 4 or combo == 14:
                directionModifier = 180
            elif combo == 12:
                directionModifier = 225
            elif combo == 8 or combo == 13:
                directionModifier = 270
            elif combo == 9:
                directionModifier = 315
            
            dPos = [
                self.frameTime * 0.025 * np.cos(np.deg2rad(self.scene.player.theta + directionModifier)),
                self.frameTime * 0.025 * np.sin(np.deg2rad(self.scene.player.theta + directionModifier)),
                0
            ]

            self.scene.move_player(dPos)
        
        # NOW HANDLE KEYS FOR ARROW
        combo = 0
        directionModifier = 0
        
        #if change theta is positive, it moves left, if it is negative, it moves right
        changeTheta = 0
        
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_UP) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_LEFT) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
            changeTheta = 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_RIGHT) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
            changeTheta = -1
        
        
        if combo > 0:
            #we need to check if W is pressed, so we actually move forward
            # so check if you are just pressing left and right
            if (combo != 2 and combo != 4 and combo != 6):
                # Now move the pyramid forward
                dPos = [
                    self.frameTime * 0.025 * np.cos(np.deg2rad(self.scene.pyramids[0].theta + directionModifier)),
                    self.frameTime * 0.025 * np.sin(np.deg2rad(self.scene.pyramids[0].theta + directionModifier)),
                    0
                ]
                #print(str(dPos))

                self.scene.move_pyramid(dPos)
                
                #ROTATE PYRAMID AROUND Y axis, as if arrow is spinning through the air
                #spin the arrow function
                rate = 20 * self.frameTime / 16.67
                self.scene.rolling_arrow(rate=rate)
            
            #HANDLES ROTATION
            #if you press left and right, don't rotate
            if (combo != 6):
                #Add rotation
                rate = 5 * self.frameTime / 16.67
                theta_reversal = -1
                theta_increment = theta_reversal * rate * changeTheta
                #no change in the pitch yet
                phi_increment = 0
                
                self.scene.spin_pyramid(theta_increment, phi_increment)
            
        
    def handleMouse(self):

        (x,y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.67
        theta_increment = rate * ((SCREEN_WIDTH / 2) - x)
        phi_increment = rate * ((SCREEN_HEIGHT / 2) - y)
        self.scene.spin_player(theta_increment, phi_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def calculateFramerate(self):

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = max(1,int(self.numFrames/delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1,framerate))
        self.numFrames += 1

    def quit(self):
        
        self.renderer.destroy()

class GraphicsEngine:


    def __init__(self):

        self.wood_texture = Material("gfx/wood.jpeg")
        self.stone_texture = Material("gfx/stone2.jpeg")
        self.cube_mesh = Mesh("models/cube.obj")
        self.pyramid_mesh = PyramidMesh()
        self.rocket_mesh = Mesh("models/rocket.obj")
        self.girl_mesh = Mesh("models/girl.obj")

        #initialise opengl
        glClearColor(0.0, 0.0, 0.0, 1)
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 50, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        # store struct of light as a position
        self.lightLocation = {
            "position": [
                glGetUniformLocation(self.shader, f"Lights[{i}].position")
                for i in range(8)
            ],
            "color": [
                glGetUniformLocation(self.shader, f"Lights[{i}].color")
                for i in range(8)
            ],
            "strength": [
                glGetUniformLocation(self.shader, f"Lights[{i}].strength")
                for i in range(8)
            ],
        }
            
        
        #get camera position location for specular component
        self.cameraPosLocation = glGetUniformLocation(self.shader, "cameraPosition")
        
        
    
    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader

    def draw_objects(self, scene):
        for pyramid in scene.pyramids:

            glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,pyramid.get_pyramid_model_matrix())
            self.stone_texture.use()
            
            #draw triangle
            glBindVertexArray(self.pyramid_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.pyramid_mesh.vertex_count)

            glFlush()

    def render(self, scene):

        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        

        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.player.position,
            target = scene.player.position + scene.player.forwards,
            up = scene.player.up, dtype = np.float32
        )
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)
        
        #update lighting information, we only have 1 light right now
        for i, light in enumerate(scene.lights):

            glUniform3fv(self.lightLocation["position"][i], 1, light.position)
            glUniform3fv(self.lightLocation["color"][i], 1, light.color)
            glUniform1f(self.lightLocation["strength"][i], light.strength)
        glUniform3fv(self.cameraPosLocation, 1, scene.player.position)
            

        self.draw_objects(scene)
        
        #print("Camera at"+ str(scene.player.position))
        #print("Facing "+ str(scene.player.position + scene.player.forwards))

        glFlush()
        
    def destroy(self):

        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)

class Mesh:


    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #enable normals
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
    
    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class PyramidMesh():
    def __init__(self):
        '''How to push triangle data to shader
        1. Set up mesh of triangle (e.g. literal vertice points)
        2. create VAO and VBO and push triangle vertice data to it
        3. create attribute pointers to tell shader which part of the buffer
        corresponds to position, colour etc in the vertex.txt file
        
        Shader:
        I want to send Location of triangle and Colour.
        That is two attributes
        Therefore in vertex.txt i need (2x) 'layout in vec3' datatypes
        
        '''

        self.setup_triangles()
        self.create_vertex_buffer_and_push()
        self.explain_to_shader_how_to_read_buffer()
    
   
    
    def setup_triangles(self):
        ''' Make a mesh of our vertices
        
        3D TRIANGLE HAS 4 triangles
        
        
        #1
        -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
        #2
        0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
        #3
        0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0,
        #4
        -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
        #5
        0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
    
        '''
        
        self.vertices = ( #now its x y z r g b S T, VN0, VN1, VN2
           
            
            #front GOOD
            -1, -1, 1, 0, 1, 0, 0.4472136, 0.89442719,
            1, -1, 1, 1, 0, 0, 0.4472136, 0.89442719,
            0, 1, 0, 0.5, 0, 0, 0.4472136, 0.89442719,

            #right side
            1, -1, 1, 0, 1, 0.9486833, 0.31622777, 0,
            1, -1, -1, 1, 1, 0.9486833, 0.31622777, 0,
            0, 1, 0, 0.5, 0, 0.9486833, 0.31622777, 0,
            
            # LEFT SIDE
            -1, -1, 1, 0, 1, -0.9486833, 0.31622777, 0,
            -1, -1, -1, 1, 1, -0.9486833, 0.31622777, 0,
            0, 1, 0, 0.5, 0, -0.9486833, 0.31622777, 0,
            
            #BACK
            -1, -1, -1, 0, 1, 0, 0.4472136, -0.89442719,
            1, -1, -1, 1, 1, 0, 0.4472136, -0.89442719,
            0, 1, 0, 0.5, 0, 0, 0.4472136, -0.89442719,
            
            # BOTTOM RIGHT
            -1, -1, 1, 0, 1, 0, -1, 0,
            1, -1, -1, 1, 1,  0, -1, 0,
            1, -1, 1, 0.5, 0,  0, -1, 0,
            
            # BOTTOM LEFT
            -1, -1, 1, 0, 1,  0, -1, 0,
            1, -1, -1, 1, 1,  0, -1, 0,
            -1, -1, -1, 0.5, 0,  0, -1, 0,
            
        )
        # convert the vertices into a numpy array
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        self.vertex_count = len(self.vertices) // 5
        
        #calculate the surface normals, using algebra
        self.calculate_surface_normal()
        

    def create_vertex_buffer_and_push(self):
        '''Create VAO and VBO
        Push our vertex object to the bush
        '''
        # vertex array object (VAO) created
        self.vao = glGenVertexArrays(1)
        # make it active
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1) # this is a vertex buffer which stores data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo) # make it active 
        upload_ready = True
        if upload_ready:
            # actually push data to array buffer
            glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

    def calculate_surface_normal(self):
        #front, right, left, back, bot right, bot left
        v1 = [-1, -1, 1]
        v2 = [1, -1, 1]
        v3 = [0, 1, 0]
        v4 = [-1, -1, -1]
        v5 = [1, -1, -1]
        sides = [
            [v1,v2,v3],
            [v2, v3, v5],
            [v1, v4, v3],
            [v4, v5, v3],
            [v1,v2,v5],
            [v1,v4,v5]
            
        ]
        faces = [
            "front","right","left","back","botright","botleft"
        ]
        index = 0
        for face in sides:
            
            A = np.subtract(face[1],face[0])
            B = np.subtract(face[2],face[0])
            Nx = A[1] * B[2] - A[2] * B[2]
            Ny = A[2] * B[0] - A[0] * B[2]
            Nz = A[0] * B[1] - A[1] * B[0]
            print(faces[index])
            print(normalize([[Nx, Ny, Nz]]))
            index += 1
        
        

    def explain_to_shader_how_to_read_buffer(self):
        '''This function uses
        glVertexAttribPointer()
        to declare which attribute in the vertex shader corresponds to which part of the buffer
        '''
        
        # LOCATION OF TRIANGLE
        #create attribute pointers. this is the contiguous blocks of triangles in memory
        attribute_index = 0
        elements_per_attribute = 3
        element_type = GL_FLOAT
        normalized = GL_FALSE #to squash the data. never do this
        stride_in_bytes = 32 #element type size in bytes * elements_per_attribute
        offset_in_bytes = 0 # how far into the datastructure is it
        
        glEnableVertexAttribArray(attribute_index)
        glVertexAttribPointer(
            attribute_index, elements_per_attribute,
            element_type, normalized,
            stride_in_bytes, ctypes.c_void_p(offset_in_bytes)
        )
        
        
        # TEXTURE OF TRIANGLE
        #create attribute pointers. this is the contiguous blocks of triangles in memory
        attribute_index = 1
        elements_per_attribute = 2
        element_type = GL_FLOAT
        normalized = GL_FALSE #to squash the data. never do this
        stride_in_bytes = 32 #element type size in bytes * elements_per_attribute
        offset_in_bytes = 12 # how far into the datastructure is it
        
        glEnableVertexAttribArray(attribute_index)
        glVertexAttribPointer(
            attribute_index, elements_per_attribute,
            element_type, normalized,
            stride_in_bytes, ctypes.c_void_p(offset_in_bytes)
        )
        
        # NORMALS
        #enable normals
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

class Material:

    
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(filepath, mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))

window = initialize_glfw()
myApp = App(window)