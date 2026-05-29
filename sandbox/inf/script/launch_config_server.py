import socket
import subprocess

sock = socket.socket()
sock.bind(("", 0))

port = sock.getsockname()[1]
sock.close()

subprocess.run(["pwd"])

subprocess.run(
    ["streamlit", "run", "sandbox/inf/script/_make_config.py", "--server.port", str(port)]
)

print(port)

