from flask import Flask
import os

app = Flask(__name__, template_folder=__path__[0] + "/templates")
print(os.path.abspath(app.template_folder))

import dexterity.monitor.app
