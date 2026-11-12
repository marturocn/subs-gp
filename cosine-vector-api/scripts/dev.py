import os
import subprocess
import sys
from dotenv import load_dotenv

def is_container_running(name: str) -> bool:
    """Devuelve True si el contenedor está corriendo."""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    return name in result.stdout.strip().split("\n")


def main() -> None:
    load_dotenv()

    # Levantar contenedores solo si no se ha deshabilitado explícitamente en el .env
    if os.getenv("SKIP_DOCKER_COMPOSE", "false").lower() == "true":
        print("Omitiendo levantamiento de contenedores con docker-compose (DB externa/local activa).")
    else:
        if not is_container_running("mysql-container"):
            print("Levantando contenedores con docker compose...")
            result = subprocess.call(["docker", "compose", "up", "-d"])
            if result != 0:
                print("Error al levantar docker-compose")
                sys.exit(result)
        else:
            print("Contenedor 'mysql-container' ya está corriendo, no se levanta de nuevo.")

    num_workers = os.getenv('API_WORKERS', '1')

    # Si se pasa el flag --uvicorn o estamos en Windows, usar Uvicorn directamente
    if "--uvicorn" in sys.argv or os.name == 'nt':
        print("Usando Uvicorn directamente" + (" (Requerido en Windows)" if os.name == 'nt' else ""))
        cmd = [
            "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", str(os.getenv('API_PORT', 8000)),
            "--reload",
        ]
        sys.exit(subprocess.call(["poetry", "run"] + cmd))

    cmd = [
        "gunicorn",
        "src.main:app",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"0.0.0.0:{os.getenv('API_PORT', 8000)}",
        "--workers", str(num_workers),
        "--reload"
    ]
    sys.exit(subprocess.call(["poetry", "run"] + cmd))


if __name__ == "__main__":
    main()
