# Monitoring

We provide an interface build as a web app that can be accessed by any common browser. No additional requirements apart from those in 
`requirements.txt` are necessary as the interface runs with a Python Flask backend. 

To start the local server (no, this is not making your computer a server, everything is happening local on your machine) you need
to run the following command in any terminal from inside the project directory:

```
env FLASK_APP=./monitor/app.py flask run
```

You should now be able to access the interface by entering `http://127.0.0.1:5000/` in your browser. The flask process should
also give you the link in the terminal.
