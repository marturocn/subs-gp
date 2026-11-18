#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error()   { echo -e "${RED}❌ $1${NC}"; }

# Auto-detect usuario no-root desde SUDO_USER (igual que services.sh); puede sobrescribirse con SERVICE_USER=usuario
if [ -z "$SERVICE_USER" ]; then
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        SERVICE_USER="$SUDO_USER"
    else
        SERVICE_USER="root"
    fi
fi

if [ "$SERVICE_USER" = "root" ]; then
    USER_HOME="/root"
else
    USER_HOME="/home/$SERVICE_USER"
fi

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --hat)
            log_info "Flag --hat reconocido (no aplica a limpieza)"
            ;;
        --no-nvidia)
            log_info "Flag --no-nvidia reconocido (no aplica a limpieza)"
            ;;
        --help|-h)
            echo "Uso: sudo ./cleaninstall.sh [--hat] [--no-nvidia]"
            echo ""
            echo "Elimina todo lo instalado por services.sh install."
            echo "Despues puede reinstalar con: sudo ./services.sh install"
            echo ""
            echo "Variables de entorno:"
            echo "  SERVICE_USER   Usuario de servicio (default: \$SUDO_USER, o root)"
            exit 0
            ;;
        *)
            log_warning "Argumento desconocido: $arg (ignorado)"
            ;;
    esac
done

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Este script debe ejecutarse como root (use sudo)"
        exit 1
    fi
}

main() {
    check_root

    echo ""
    log_info "=== cleaninstall.sh: Limpieza de instalacion de servicios subs-gp ==="
    log_info "SERVICE_USER: $SERVICE_USER"
    log_info "USER_HOME:    $USER_HOME"
    echo ""

    # Track what was removed and what was skipped
    REMOVED=()
    SKIPPED=()

    # -------------------------------------------------------------------------
    # 1. Stop, disable, and remove systemd services
    # -------------------------------------------------------------------------
    log_info "Procesando servicios systemd..."

    for SVC in clip-vector-subs-gp cosine-vector-subs-gp; do
        SERVICE_FILE="/etc/systemd/system/${SVC}.service"

        log_info "Deteniendo y deshabilitando servicio: $SVC"
        systemctl stop "$SVC" 2>/dev/null || true
        systemctl disable "$SVC" 2>/dev/null || true

        if [ -f "$SERVICE_FILE" ]; then
            rm -f "$SERVICE_FILE"
            log_success "Eliminado archivo de servicio: $SERVICE_FILE"
            REMOVED+=("$SERVICE_FILE")
        else
            log_warning "Archivo de servicio no existe: $SERVICE_FILE (omitido)"
            SKIPPED+=("$SERVICE_FILE")
        fi
    done

    # -------------------------------------------------------------------------
    # 2. Reload systemd daemon
    # -------------------------------------------------------------------------
    log_info "Recargando daemon de systemd..."
    systemctl daemon-reload
    log_success "systemctl daemon-reload completado"

    # -------------------------------------------------------------------------
    # 3. Remove service install directories
    # -------------------------------------------------------------------------
    log_info "Eliminando directorios de instalacion de servicios..."

    for DIR in \
        "/opt/subs-gp/clip-vector-api" \
        "/opt/subs-gp/cosine-vector-api" \
        "/opt/subs-gp/CLIP_ENVIRONMENT"; do
        if [ -d "$DIR" ]; then
            rm -rf "$DIR"
            log_success "Eliminado: $DIR"
            REMOVED+=("$DIR")
        else
            log_warning "No existe: $DIR (omitido)"
            SKIPPED+=("$DIR")
        fi
    done

    # -------------------------------------------------------------------------
    # 4. Remove state files
    # -------------------------------------------------------------------------
    log_info "Eliminando archivos de estado..."

    for STATE_FILE in \
        "/opt/subs-gp/.prod_install_complete" \
        "/opt/subs-gp/nvidia_install_state" \
        "/opt/subs-gp/nvidia_reboot_count" \
        "/opt/subs-gp/nvidia_fallback_tried"; do
        if [ -f "$STATE_FILE" ]; then
            rm -f "$STATE_FILE"
            log_success "Eliminado: $STATE_FILE"
            REMOVED+=("$STATE_FILE")
        else
            log_warning "No existe: $STATE_FILE (omitido)"
            SKIPPED+=("$STATE_FILE")
        fi
    done

    # -------------------------------------------------------------------------
    # 5. Remove /opt/subs-gp/ if empty
    # -------------------------------------------------------------------------
    if [ -d "/opt/subs-gp" ]; then
        if [ -z "$(ls -A /opt/subs-gp 2>/dev/null)" ]; then
            rmdir "/opt/subs-gp"
            log_success "Eliminado directorio vacio: /opt/subs-gp"
            REMOVED+=("/opt/subs-gp")
        else
            log_warning "/opt/subs-gp no esta vacio, no se elimina. Contenido restante:"
            ls -la /opt/subs-gp
        fi
    else
        log_warning "No existe: /opt/subs-gp (omitido)"
        SKIPPED+=("/opt/subs-gp")
    fi

    # -------------------------------------------------------------------------
    # 6. Remove pyenv for SERVICE_USER
    # -------------------------------------------------------------------------
    log_info "Buscando y eliminando pyenv en todos los usuarios (/root y /home/*) ..."

    # Buscar directorios que se llamen "pyenv" o ".pyenv"
    while IFS= read -r PYENV_DIR; do
        if [ -d "$PYENV_DIR" ]; then
            rm -rf "$PYENV_DIR"
            log_success "Eliminado directorio pyenv: $PYENV_DIR"
            REMOVED+=("$PYENV_DIR")
        fi
    done < <(find /root /home -maxdepth 5 -type d \( -name ".pyenv" -o -name "pyenv" \) 2>/dev/null || true)

    # -------------------------------------------------------------------------
    # 7. Remove Poetry cache directories for SERVICE_USER
    # -------------------------------------------------------------------------
    log_info "Buscando y eliminando cachés y binarios de Poetry en todos los usuarios (/root y /home/*)..."

    # Buscar directorios de cache/config de poetry
    while IFS= read -r POETRY_DIR; do
        if [ -d "$POETRY_DIR" ]; then
            rm -rf "$POETRY_DIR"
            log_success "Eliminado directorio poetry: $POETRY_DIR"
            REMOVED+=("$POETRY_DIR")
        fi
    done < <(find /root /home -maxdepth 5 -type d \( -name "pypoetry" -o -name ".poetry" \) 2>/dev/null || true)

    # Buscar binarios de poetry
    while IFS= read -r POETRY_BIN; do
        if [ -f "$POETRY_BIN" ] && [ -x "$POETRY_BIN" ]; then
            rm -f "$POETRY_BIN"
            log_success "Eliminado binario poetry: $POETRY_BIN"
            REMOVED+=("$POETRY_BIN")
        fi
    done < <(find /root /home -maxdepth 5 -type f -name "poetry" 2>/dev/null || true)

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  RESUMEN DE LIMPIEZA${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    if [ ${#REMOVED[@]} -gt 0 ]; then
        echo -e "${GREEN}Eliminado:${NC}"
        for item in "${REMOVED[@]}"; do
            echo -e "  ${GREEN}✅ $item${NC}"
        done
    else
        echo -e "${YELLOW}No se elimino ningún elemento.${NC}"
    fi

    echo ""
    echo -e "${YELLOW}NO tocado (se mantiene intacto):${NC}"
    echo -e "  ${YELLOW}⚠️  Drivers NVIDIA${NC}"
    echo -e "  ${YELLOW}⚠️  CUDA Toolkit${NC}"
    echo -e "  ${YELLOW}⚠️  Paquetes del sistema${NC}"
    echo -e "  ${YELLOW}⚠️  Otros datos en /opt/ fuera de /opt/subs-gp/${NC}"

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Ejecute sudo ./services.sh install para reinstalar${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

main
