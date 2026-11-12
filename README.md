# Vectorización de imágenes

Dos servicios systemd para procesamiento de imágenes con IA y búsqueda vectorial:

| Servicio             | Nombre systemd               | Proyecto            | Ruta instalación                  | Puerto |
| -------------------- | ---------------------------- | ------------------- | --------------------------------- | ------ |
| CLIP Embedding (GPU) | `clip-vector-subs-gp`        | `clip-vector-api`   | `/opt/subs-gp/clip-vector-api`    | 8200   |
| Cosine Similarity    | `cosine-vector-subs-gp`      | `cosine-vector-api` | `/opt/subs-gp/cosine-vector-api`  | 8300   |

---

## Instalación rápida (RHEL EC2 g4dn.xlarge)

```bash
# RHEL 9 con GPU NVIDIA Tesla T4 (crea el entorno virtual en /opt/subs-gp/CLIP_ENVIRONMENT)
cd /opt/subs-gp/clip-vector-api && ./prod.sh --hat --env-false
cd /opt/subs-gp/cosine-vector-api && ./prod.sh --hat --env-false

# Copiar y activar servicios systemd
sudo cp /opt/subs-gp/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now clip-vector-subs-gp cosine-vector-subs-gp
```

---

## services.sh — Gestión de servicios

```bash
sudo ./services.sh <comando> [opciones]
```

### Comandos

| Comando        | Descripción                                                              |
| -------------- | ------------------------------------------------------------------------ |
| `install`      | Copia archivos, genera unit files e inicia servicios en `/opt/subs-gp/`  |
| `start`        | Inicia todos los servicios                                               |
| `stop`         | Detiene todos los servicios                                              |
| `restart`      | Reinicia todos los servicios                                             |
| `status [svc]` | Estado de todos o de un servicio específico                              |
| `uninstall`    | Detiene, deshabilita y elimina los unit files (no borra `/opt/subs-gp/`) |
| `logs <svc>`   | Sigue los logs en tiempo real (`journalctl -f`)                          |
| `help`         | Muestra la ayuda                                                         |

### Opciones de instalación

| Opción        | Efecto                                                                                |
| ------------- | ------------------------------------------------------------------------------------- |
| `--hat`       | Usa `dnf` en lugar de `apt` (RHEL/CentOS/Rocky)                                       |
| `--no-nvidia` | Omite instalación de drivers NVIDIA/CUDA en clip-vector-subs-gp                       |
| `--env-true`  | Virtualenv por proyecto en `.venv/`                                                   |
| `--env-false` | Virtualenv compartido en `CLIP_VENV` (`/opt/subs-gp/CLIP_ENVIRONMENT`)                |

---

## cleaninstall.sh — Limpieza completa de subs-gp

Elimina toda la instalación de `subs-gp` para reiniciar desde cero en caso de ser necesario:

```bash
sudo ./cleaninstall.sh
```

**Lo que elimina:**

| Elemento                 | Ruta                                                                   |
| ------------------------ | ---------------------------------------------------------------------- |
| Servicio clip            | `/etc/systemd/system/clip-vector-subs-gp.service`                      |
| Servicio cosine          | `/etc/systemd/system/cosine-vector-subs-gp.service`                    |
| Directorio clip          | `/opt/subs-gp/clip-vector-api/`                                        |
| Directorio cosine        | `/opt/subs-gp/cosine-vector-api/`                                      |
| Entorno Python (Venv)    | `/opt/subs-gp/CLIP_ENVIRONMENT/`                                       |
| Estado de instalación    | `/opt/subs-gp/.prod_install_complete`                                  |
| Estado NVIDIA            | `/opt/subs-gp/nvidia_install_state`, `/opt/subs-gp/nvidia_reboot_count` |

**Lo que NO toca:**

- Drivers NVIDIA y CUDA del sistema
- Paquetes base del sistema operativo
- Directorio de la versión previa en `/opt/artg/`

---

## prod.sh — Script de arranque de cada servicio

```bash
cd /opt/subs-gp/clip-vector-api
./prod.sh --hat --env-false

cd /opt/subs-gp/cosine-vector-api
./prod.sh --hat --env-false
```

### Modos de entorno Python

| Flag          | Virtualenv                                                        | Instalación                       | Ejecución             |
| ------------- | ----------------------------------------------------------------- | --------------------------------- | --------------------- |
| _(ninguno)_   | `virtualenvs.create false` → site-packages de pyenv               | `poetry install`                  | `poetry run prod`     |
| `--env-true`  | `.venv/` en el directorio del proyecto                            | `poetry install` en cada arranque | `poetry run prod`     |
| `--env-false` | `CLIP_VENV` del `.env` (`/opt/subs-gp/CLIP_ENVIRONMENT`)          | `poetry install` en ese venv      | `$CLIP_VENV/bin/prod` |

---

## Diagnóstico rápido

```bash
# Ver estado de ambos servicios
sudo systemctl status clip-vector-subs-gp cosine-vector-subs-gp

# Ver logs en tiempo real
sudo journalctl -u clip-vector-subs-gp -f
sudo journalctl -u cosine-vector-subs-gp -f

# Verificar GPU
nvidia-smi

# Verificar puertos en escucha
sudo ss -tulpn | grep -E '8200|8300'
```

---

## Rutas importantes

```
/opt/subs-gp/
├── clip-vector-api/              # Microservicio de inferencia CLIP/DINO (Puerto 8200)
│   ├── prod.sh
│   ├── .env
│   └── .python-version
├── cosine-vector-api/            # Microservicio de búsqueda FAISS (Puerto 8300)
│   ├── prod.sh
│   └── .env
├── CLIP_ENVIRONMENT/             # Entorno virtual Python compartido
└── .prod_install_complete        # Marker de instalación completada

/etc/systemd/system/
├── clip-vector-subs-gp.service   # Unidad systemd para CLIP
└── cosine-vector-subs-gp.service # Unidad systemd para Cosine
```
