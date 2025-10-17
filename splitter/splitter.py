import socket
import pickle
import numpy as np
import uuid

# Configuration
HOST = '0.0.0.0'  # Bind to all interfaces in Docker
PORT = 5557       # Port to send tasks to Multiplication Service
THRESHOLD = 2     # Minimum sub-matrix size (for testing)

# Function to validate matrices
def validate_matrices(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same dimensions")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Only square matrices are supported")
    if not (np.issubdtype(A.dtype, np.number) and np.issubdtype(B.dtype, np.number)):
        raise ValueError("Matrices must be numeric")

# Recursive divide-and-conquer split
def recursive_split(A, B, tasks):
    n = A.shape[0]
    if n <= THRESHOLD:
        task_id = str(uuid.uuid4())
        tasks.append({'task_id': task_id, 'subA': A, 'subB': B})
        return

    mid = n // 2
    A_quads = [A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]]
    B_quads = [B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]]

    for a_sub, b_sub in zip(A_quads, B_quads):
        recursive_split(a_sub, b_sub, tasks)

# Function to send tasks over TCP
def send_tasks(tasks, host, port):
    for task in tasks:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                s.sendall(pickle.dumps(task))  # Serialize Python object
                print(f"Sent task {task['task_id']} to Multiplication Service")
            except Exception as e:
                print(f"Failed to send task {task['task_id']}: {e}")

def main():
    # Example: generate random 4x4 matrices (replace with file input if needed)
    n = 4
    A = np.random.randint(0, 10, (n, n))
    B = np.random.randint(0, 10, (n, n))
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    validate_matrices(A, B)

    tasks = []
    recursive_split(A, B, tasks)
    print(f"Total tasks generated: {len(tasks)}")

    send_tasks(tasks, HOST, PORT)

if __name__ == "__main__":
    main()
