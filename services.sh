#!/bin/bash

set -e

# =============================================================================
# SCRIPT DE CONFIGURACIÓN Y DESPLIEGUE DE SERVICIOS EN SYSTEMD
# =============================================================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flag para omitir NVIDIA
SKIP_NVIDIA=false
# Flag para RHEL (Red Hat Enterprise Linux)
RHEL_MODE=false
ENV_TRUE_MODE=false
ENV_FALSE_MODE=false

# Directorio base donde están los proyectos
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuración de servicios
# Formato: "nombre_servicio:directorio_proyecto:ruta_instalacion:requiere_gpu"
declare -A SERVICES=(
    ["clip-vector-subs-gp"]="clip-vector-api:/opt/subs-gp/clip-vector-api:true"
    ["cosine-vector-subs-gp"]="cosine-vector-api:/opt/subs-gp/cosine-vector-api:false"
)

# Alias para nombres de servicios
declare -A SERVICE_ALIASES=(
    ["clip"]="clip-vector-subs-gp"
    ["embedding"]="clip-vector-subs-gp"
    ["ei"]="clip-vector-subs-gp"
    ["cosine"]="cosine-vector-subs-gp"
    ["vector"]="cosine-vector-subs-gp"
    ["vs"]="cosine-vector-subs-gp"
)

# Auto-detect usuario no-root desde SUDO_USER; puede sobrescribirse con SERVICE_USER=usuario
if [ -z "$SERVICE_USER" ]; then
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        SERVICE_USER="$SUDO_USER"
    else
        SERVICE_USER="root"
    fi
fi
# Home directory del service user
if [ "$SERVICE_USER" = "root" ]; then
    USER_HOME="/root"
else
    USER_HOME="/home/$SERVICE_USER"
fi

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Muestra el estado de un servicio y distingue "arrancando" de "fallando".
# El primer arranque de clip-vector-subs-gp instala drivers/CUDA (hasta 30 min) en estado
# activating; eso no es un fallo.
print_service_health() {
    local service_name="$1"
    systemctl status "${service_name}.service" --no-pager 2>/dev/null || true

    local state
    state=$(systemctl is-active "${service_name}.service" 2>/dev/null || true)
    case "$state" in
        active)
            ;;
        activating|reloading)
            echo ""
            log_info "⏳ ${service_name} está arrancando. El primer arranque instala dependencias y puede tardar hasta 30 min."
            log_info "   Sigue el avance con: sudo journalctl -u ${service_name} -f"
            ;;
        *)
            echo ""
            log_warning "El servicio está fallando (estado: ${state:-desconocido}). Mostrando últimas líneas del log:"
            echo ""
            journalctl -u "${service_name}.service" -n 20 --no-pager 2>/dev/null || true
            ;;
    esac
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Este script debe ejecutarse con sudo"
        echo "Uso: sudo ./services.sh [install|start|stop|restart|status|uninstall]"
        exit 1
    fi
}

# =============================================================================
# GENERACIÓN DE ARCHIVOS DE SERVICIO
# =============================================================================

generate_gpu_service() {
    local service_name=$1
    local install_path=$2
    local skip_nvidia=$3
    
    local exec_cmd="${install_path}/prod.sh"
    local extra_args=""
    
    if [ "$skip_nvidia" = "true" ]; then
        extra_args="$extra_args --no-nvidia"
    fi
    
    if [ "$RHEL_MODE" = "true" ]; then
        extra_args="$extra_args --hat"
    fi

    if [ "$ENV_TRUE_MODE" = "true" ]; then
        extra_args="$extra_args --env-true"
    fi

    if [ "$ENV_FALSE_MODE" = "true" ]; then
        extra_args="$extra_args --env-false"
    fi

    if [ -n "$extra_args" ]; then
        exec_cmd="${install_path}/prod.sh$extra_args"
    fi

    cat << EOF
[Unit]
Description=${service_name} - Embedding Image Service (GPU)
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
User=${SERVICE_USER}
WorkingDirectory=${install_path}

# CUDA y TensorRT
Environment="PATH=${USER_HOME}/.pyenv/shims:${USER_HOME}/.pyenv/bin:${USER_HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/usr/local/cuda/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYENV_ROOT=${USER_HOME}/.pyenv"
Environment="HOME=${USER_HOME}"

# Permitir acceso a la GPU desde systemd
DeviceAllow=/dev/nvidiactl
DeviceAllow=/dev/nvidia0
DeviceAllow=/dev/nvidia-uvm
DeviceAllow=/dev/nvidia-uvm-tools
DevicePolicy=auto

# Instalar dependencias del sistema como root antes de arrancar
ExecStartPre=+/bin/bash -c 'if command -v dnf >/dev/null 2>&1; then dnf install -y bzip2-devel openssl-devel readline-devel sqlite-devel libffi-devel zlib-devel xz-devel gcc make patch libuuid-devel 2>/dev/null || true; elif command -v apt-get >/dev/null 2>&1; then DEBIAN_FRONTEND=noninteractive apt-get install -y libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libffi-dev zlib1g-dev build-essential 2>/dev/null || true; fi'

# Ejecutar el servicio
ExecStart=${exec_cmd}

Restart=always
RestartSec=5
TimeoutStartSec=0
LimitNOFILE=65535

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

generate_standard_service() {
    local service_name=$1
    local install_path=$2
    
    local exec_cmd="${install_path}/prod.sh"
    local extra_args=""

    if [ "$RHEL_MODE" = "true" ]; then
        extra_args="$extra_args --hat"
    fi

    if [ "$ENV_TRUE_MODE" = "true" ]; then
        extra_args="$extra_args --env-true"
    fi

    if [ "$ENV_FALSE_MODE" = "true" ]; then
        extra_args="$extra_args --env-false"
    fi

    if [ -n "$extra_args" ]; then
        exec_cmd="${install_path}/prod.sh$extra_args"
    fi

    cat << EOF
[Unit]
Description=${service_name} - Vector Similarity Service
After=network.target

[Service]
User=${SERVICE_USER}
WorkingDirectory=${install_path}
Environment="PATH=${USER_HOME}/.pyenv/shims:${USER_HOME}/.pyenv/bin:${USER_HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="PYENV_ROOT=${USER_HOME}/.pyenv"
Environment="HOME=${USER_HOME}"

# Instalar dependencias del sistema como root antes de arrancar
ExecStartPre=+/bin/bash -c 'if command -v dnf >/dev/null 2>&1; then dnf install -y bzip2-devel openssl-devel readline-devel sqlite-devel libffi-devel zlib-devel xz-devel gcc make patch libuuid-devel 2>/dev/null || true; elif command -v apt-get >/dev/null 2>&1; then DEBIAN_FRONTEND=noninteractive apt-get install -y libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libffi-dev zlib1g-dev build-essential 2>/dev/null || true; fi'

ExecStart=${exec_cmd}
Restart=always
RestartSec=5
TimeoutStartSec=0
LimitNOFILE=65535

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

# =============================================================================
# INSTALACIÓN DE SERVICIOS
# =============================================================================

install_service() {
    local service_name=$1
    local project_dir=$2
    local install_path=$3
    local requires_gpu=$4
    
    local source_path="${SCRIPT_DIR}/${project_dir}"
    local service_file="/etc/systemd/system/${service_name}.service"
    
    log_info "Instalando servicio: ${service_name}"
    
    # Verificar que el proyecto existe
    if [ ! -d "$source_path" ]; then
        log_error "Directorio del proyecto no encontrado: ${source_path}"
        return 1
    fi
    
    # Crear directorio de instalación si no existe
    if [ ! -d "$install_path" ]; then
        log_info "Creando directorio: ${install_path}"
        mkdir -p "$install_path"
    fi
    
    # Copiar archivos del proyecto
    log_info "Copiando archivos de ${source_path} a ${install_path}..."
    rsync -av --exclude='.git' \
              --exclude='__pycache__' \
              --exclude='.pytest_cache' \
              --exclude='*.pyc' \
              --exclude='.venv' \
              --exclude='node_modules' \
              --exclude='.mypy_cache' \
              "${source_path}/" "${install_path}/"
    
    # Ajustar permisos
    chown -R "${SERVICE_USER}:${SERVICE_USER}" "$install_path"
    chmod +x "${install_path}/prod.sh"
    
    # Generar archivo de servicio
    log_info "Generando archivo de servicio: ${service_file}"
    if [ "$requires_gpu" = "true" ]; then
        generate_gpu_service "$service_name" "$install_path" "$SKIP_NVIDIA" > "$service_file"
    else
        generate_standard_service "$service_name" "$install_path" > "$service_file"
    fi
    
    log_success "Servicio ${service_name} instalado correctamente"
}

install_all_services() {
    log_info "Instalando todos los servicios..."
    
    # Crear directorio compartido CLIP_ENVIRONMENT si no existe
    local clip_env_path="/opt/subs-gp/CLIP_ENVIRONMENT"
    if [ ! -d "$clip_env_path" ]; then
        log_info "Creando directorio compartido: ${clip_env_path}"
        mkdir -p "$clip_env_path"
        chown -R "${SERVICE_USER}:${SERVICE_USER}" "$clip_env_path"
    fi
    
    for service_name in "${!SERVICES[@]}"; do
        IFS=':' read -r project_dir install_path requires_gpu <<< "${SERVICES[$service_name]}"
        install_service "$service_name" "$project_dir" "$install_path" "$requires_gpu"
    done
    
    # Recargar systemd
    log_info "Recargando systemd..."
    systemctl daemon-reload
    
    log_success "Todos los servicios instalados correctamente"
}

# =============================================================================
# CONTROL DE SERVICIOS
# =============================================================================

start_services() {
    log_info "Iniciando servicios..."
    
    for service_name in "${!SERVICES[@]}"; do
        log_info "Iniciando ${service_name}..."
        # enable primero y start sin bloquear: el primer arranque de clip-vector-subs-gp instala
        # drivers/CUDA (puede tardar 30 min o cortar D-Bus) y no debe frenar al resto
        systemctl enable "${service_name}.service"
        systemctl start --no-block "${service_name}.service"
        log_success "${service_name} habilitado y arrancando (ver: ./services.sh logs)"
    done
    
    log_success "Todos los servicios iniciados"
}

stop_services() {
    log_info "Deteniendo servicios..."
    
    for service_name in "${!SERVICES[@]}"; do
        log_info "Deteniendo ${service_name}..."
        systemctl stop "${service_name}.service" 2>/dev/null || true
        log_success "${service_name} detenido"
    done
    
    log_success "Todos los servicios detenidos"
}

restart_services() {
    log_info "Reiniciando servicios..."
    
    for service_name in "${!SERVICES[@]}"; do
        log_info "Reiniciando ${service_name}..."
        systemctl restart "${service_name}.service"
        log_success "${service_name} reiniciado"
    done
    
    log_success "Todos los servicios reiniciados"
}

status_services() {
    local lines=${1:-0}
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                    ESTADO DE LOS SERVICIOS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    for service_name in "${!SERVICES[@]}"; do
        IFS=':' read -r project_dir install_path requires_gpu <<< "${SERVICES[$service_name]}"
        local service_file="/etc/systemd/system/${service_name}.service"
        
        echo "───────────────────────────────────────────────────────────────"
        echo "Servicio: ${service_name}"
        echo "───────────────────────────────────────────────────────────────"
        
        if [ ! -f "$service_file" ]; then
            log_error "No instalado"
            echo ""
            echo "  📁 Archivo de servicio: $service_file (no existe)"
            echo "  📂 Ruta de instalación: $install_path"
            echo "  📦 Proyecto origen:     ${SCRIPT_DIR}/${project_dir}"
            echo ""
            echo "  💡 Para instalar, ejecute:"
            echo "     sudo ./services.sh install"
        else
            print_service_health "${service_name}"
            
            # Si se especificaron líneas adicionales, mostrar logs
            if [ "$lines" -gt 0 ]; then
                echo ""
                echo "📜 Últimas $lines líneas del log:"
                echo ""
                journalctl -u "${service_name}.service" -n "$lines" --no-pager 2>/dev/null || true
            fi
        fi
        echo ""
    done
}

# =============================================================================
# DESINSTALACIÓN
# =============================================================================

uninstall_services() {
    log_warning "Desinstalando todos los servicios..."
    
    for service_name in "${!SERVICES[@]}"; do
        log_info "Desinstalando ${service_name}..."
        
        # Detener y deshabilitar servicio
        systemctl stop "${service_name}.service" 2>/dev/null || true
        systemctl disable "${service_name}.service" 2>/dev/null || true
        
        # Eliminar archivo de servicio
        rm -f "/etc/systemd/system/${service_name}.service"
        
        log_success "${service_name} desinstalado"
    done
    
    # Recargar systemd
    systemctl daemon-reload
    
    log_success "Todos los servicios desinstalados"
    log_warning "Nota: Los archivos en /opt/subs-gp/ NO fueron eliminados"
}

# =============================================================================
# VER LOGS
# =============================================================================

resolve_service_name() {
    local input=$1
    # Si es un alias, resolver al nombre real
    if [[ -n "${SERVICE_ALIASES[$input]}" ]]; then
        echo "${SERVICE_ALIASES[$input]}"
    else
        echo "$input"
    fi
}

show_logs() {
    local service_input=$1
    local lines=${2:-100}
    
    if [ -z "$service_input" ]; then
        log_error "Debe especificar el nombre del servicio"
        echo "Servicios disponibles: ${!SERVICES[*]}"
        echo "Alias disponibles: ${!SERVICE_ALIASES[*]}"
        exit 1
    fi
    
    local service_name
    service_name=$(resolve_service_name "$service_input")
    
    journalctl -u "${service_name}.service" -f -n "$lines"
}

show_service_status() {
    local service_input=$1
    local lines=${2:-0}
    
    if [ -z "$service_input" ]; then
        # Mostrar todos los servicios
        status_services "$lines"
        return
    fi
    
    local service_name
    service_name=$(resolve_service_name "$service_input")
    local service_file="/etc/systemd/system/${service_name}.service"
    
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "Servicio: ${service_name}"
    echo "───────────────────────────────────────────────────────────────"
    
    if [ ! -f "$service_file" ]; then
        log_error "No instalado"
        echo ""
        echo "  📁 Archivo de servicio: $service_file (no existe)"
        echo ""
        echo "  💡 Para instalar, ejecute:"
        echo "     sudo ./services.sh install"
    else
        print_service_health "${service_name}"
        
        # Si se especificaron líneas adicionales
        if [ "$lines" -gt 0 ]; then
            echo ""
            echo "📜 Últimas $lines líneas del log:"
            echo ""
            journalctl -u "${service_name}.service" -n "$lines" --no-pager 2>/dev/null || true
        fi
    fi
    echo ""
}

# =============================================================================
# AYUDA
# =============================================================================

show_help() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "         SCRIPT DE GESTIÓN DE SERVICIOS SYSTEMD"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Uso: sudo ./services.sh [comando] [opciones]"
    echo ""
    echo "Comandos disponibles:"
    echo "  install        Instala y configura todos los servicios"
    echo "  start          Inicia todos los servicios"
    echo "  stop           Detiene todos los servicios"
    echo "  restart        Reinicia todos los servicios"
    echo "  status [svc]   Muestra el estado de todos o un servicio específico"
    echo "  uninstall      Desinstala todos los servicios"
    echo "  logs <svc>     Muestra logs del servicio especificado"
    echo "  help           Muestra esta ayuda"
    echo ""
    echo "Servicios configurados:"
    for service_name in "${!SERVICES[@]}"; do
        IFS=':' read -r project_dir install_path requires_gpu <<< "${SERVICES[$service_name]}"
        echo "  - ${service_name}: ${project_dir} -> ${install_path}"
        if [ "$requires_gpu" = "true" ]; then
            echo "    (con soporte GPU)"
        fi
    done
    echo ""
    echo "Alias de servicios:"
    echo "  clip, embedding, ei  -> clip-vector-subs-gp (clip-vector-api)"
    echo "  cosine, vector, vs   -> cosine-vector-subs-gp (cosine-vector-api)"
    echo ""
    echo "Variables de entorno:"
    echo "  SERVICE_USER  Usuario para ejecutar servicios (default: ubuntu)"
    echo ""
    echo "Opciones:"
    echo "  --no-nvidia   Omitir instalación de NVIDIA/CUDA en prod.sh de clip-vector-api"
    echo "  --hat         Usar comandos de RHEL/CentOS (dnf) en lugar de Ubuntu (apt)"
    echo "  --env-true    Crear virtualenv por proyecto (no compartido) con virtualenvs.create true"
    echo "  --env-false   Instalar en CLIP_VENV y ejecutar \$CLIP_VENV/bin/prod directamente"
    echo ""
    echo "Ejemplos:"
    echo "  sudo ./services.sh install              # Instala todos los servicios"
    echo "  sudo ./services.sh install --no-nvidia  # Instala sin setup de NVIDIA"
    echo "  sudo ./services.sh install --hat        # Instala en RHEL/CentOS"
    echo "  sudo ./services.sh install --hat --env-true  # Instala con venv aislado por proyecto"
    echo "  sudo ./services.sh install --hat --env-false # Instala con venv en CLIP_VENV"
    echo "  sudo ./services.sh start                # Inicia todos los servicios"
    echo "  sudo ./services.sh status               # Ver estado de todos"
    echo "  sudo ./services.sh status clip          # Ver estado de clip-vector-api"
    echo "  sudo ./services.sh status vs            # Ver estado de cosine-vector-api"
    echo "  sudo ./services.sh logs clip-vector-subs-gp  # Ver logs de clip-vector-api"
    echo "  sudo ./services.sh logs vs              # Ver logs de cosine-vector-api"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local command=""
    local log_arg1=""
    local log_arg2=""
    local status_arg=""
    local status_lines="0"
    
    # Procesar TODOS los argumentos primero
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-nvidia)
                SKIP_NVIDIA=true
                shift
                ;;
            --hat)
                RHEL_MODE=true
                shift
                ;;
            --env-true)
                ENV_TRUE_MODE=true
                shift
                ;;
            --env-false)
                ENV_FALSE_MODE=true
                shift
                ;;
            logs)
                command="logs"
                shift
                log_arg1="${1:-}"
                log_arg2="${2:-}"
                break
                ;;
            status)
                command="status"
                shift
                status_arg="${1:-}"
                status_lines="${2:-0}"
                break
                ;;
            install|start|stop|restart|uninstall|help|--help|-h)
                command=$1
                shift
                ;;
            *)
                if [ -z "$command" ]; then
                    command=$1
                fi
                shift
                ;;
        esac
    done
    
    # Mostrar usuario y home detectados
    log_info "SERVICE_USER=${SERVICE_USER}  USER_HOME=${USER_HOME}"

    # Mostrar flags activas
    if [ "$SKIP_NVIDIA" = "true" ]; then
        log_info "Modo --no-nvidia activado"
    fi
    
    if [ "$RHEL_MODE" = "true" ]; then
        log_info "Modo --hat (RHEL) activado"
    fi

    if [ "$ENV_TRUE_MODE" = "true" ]; then
        log_info "Modo --env-true activado (virtualenv por proyecto)"
    fi

    if [ "$ENV_FALSE_MODE" = "true" ]; then
        log_info "Modo --env-false activado (virtualenv en CLIP_VENV)"
    fi

    # Si no hay comando, usar help
    command=${command:-help}
    
    case $command in
        install)
            check_root
            install_all_services
            start_services
            status_services
            ;;
        start)
            check_root
            start_services
            ;;
        stop)
            check_root
            stop_services
            ;;
        restart)
            check_root
            restart_services
            ;;
        status)
            show_service_status "$status_arg" "$status_lines"
            ;;
        uninstall)
            check_root
            uninstall_services
            ;;
        logs)
            show_logs "$log_arg1" "$log_arg2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Comando desconocido: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
