import socket
import pickle

HOST = '0.0.0.0'
PORT = 5557

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Dummy worker listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        data = b''
        while True:
            part = conn.recv(4096)
            if not part:
                break
            data += part
        task = pickle.loads(data)
        print(f"\n📦 Received task from {addr}")
        print(f"Task ID: {task['task_id']}")
        print(f"Submatrix A:\n{task['subA']}")
        print(f"Submatrix B:\n{task['subB']}")
        conn.close()
