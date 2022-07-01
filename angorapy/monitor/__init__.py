from flask import Flask
import os

app = Flask(__name__, template_folder=__path__[0] + "/templates")

import angorapy.monitor.app
