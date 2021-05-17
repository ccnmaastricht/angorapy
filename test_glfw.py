import glfw

glfw.init()
if glfw.init():
    print("Successfully initialized GLFW")
else:
    print("Failed to initialize GLFW")
