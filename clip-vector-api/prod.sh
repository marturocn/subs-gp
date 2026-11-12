#!/bin/bash

set -e

# Log file para debugging
INSTALL_LOG="/var/log/nvidia-install.log"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# PROCESAMIENTO DE ARGUMENTOS
# =============================================================================

SKIP_NVIDIA=false
RHEL_MODE=false
FULL_INSTALL_STATE_FILE="/opt/subs-gp/.prod_install_complete"
ENV_TRUE=false
ENV_FALSE=false


while [[ $# -gt 0 ]]; do
    case $1 in
        --no-nvidia)
            SKIP_NVIDIA=true
            echo "ℹ️  Omitiendo instalación de NVIDIA/CUDA (--no-nvidia)"
            shift
            ;;
        --hat)
            RHEL_MODE=true
            echo "ℹ️  Modo RHEL activado (--hat)"
            shift
            ;;
        --env-true)
            ENV_TRUE=true
            echo "ℹ️  Modo --env-true activado (virtualenv por proyecto)"
            shift
            ;;
        --env-false)
            ENV_FALSE=true
            echo "ℹ️  Modo --env-false activado (virtualenv en CLIP_VENV)"
            shift
            ;;
        -h|--help)
            echo "Uso: ./prod.sh [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --no-nvidia    Omitir instalación de drivers NVIDIA/CUDA/TensorRT"
            echo "  --hat          Usar comandos RHEL/CentOS (dnf) en lugar de Ubuntu (apt)"
            echo "  --env-true     Usar virtualenv por proyecto (poetry virtualenvs.in-project)"
            echo "  --env-false    Usar virtualenv en CLIP_VENV (leído de .env)"
            echo "  -h, --help     Mostrar esta ayuda"
            exit 0
            ;;
        *)
            echo "⚠️  Opción desconocida: $1"
            echo "Use --help para ver las opciones disponibles"
            exit 1
            ;;
    esac
done

PYTHON_VERSION=$(cat .python-version | tr -d '[:space:]')

# =============================================================================
# VERIFICACIÓN DE CONFIGURACIÓN INICIAL
# =============================================================================

echo "🔍 Verificando configuración del proyecto..."

# Verificar si existe archivo .env
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  Archivo .env no encontrado en el directorio actual"
    echo "   Creando .env a partir de .env.prod..."
    echo ""
    
    if [ -f .env.prod ]; then
        cp .env.prod .env
        echo "✅ Archivo .env creado exitosamente a partir de .env.prod"
    else
        echo "❌ Error: Ni .env ni .env.prod encontrados"
        echo "   Por favor cree el archivo .env antes de continuar"
        exit 1
    fi
else
    echo "✅ Archivo .env encontrado"
fi

echo "🔍 Verificando dependencias del sistema..."

# =============================================================================
# FUNCIONES DE INSTALACIÓN DE PAQUETES (UBUNTU vs RHEL)
# =============================================================================

pkg_update() {
    if [ "$RHEL_MODE" = "true" ]; then
        sudo dnf check-update -y || true
    else
        sudo apt-get update -y
    fi
}

pkg_install() {
    if [ "$RHEL_MODE" = "true" ]; then
        sudo dnf install -y "$@"
    else
        sudo apt-get install -y "$@"
    fi
}

pkg_clean() {
    if [ "$RHEL_MODE" = "true" ]; then
        sudo dnf clean all
        sudo dnf autoremove -y || true
    else
        sudo apt-get clean
        sudo apt-get autoremove -y
    fi
}

pkg_check_installed() {
    local pkg_name=$1
    if [ "$RHEL_MODE" = "true" ]; then
        rpm -q "$pkg_name" &> /dev/null
    else
        dpkg -l | grep -q "$pkg_name"
    fi
}

# =============================================================================
# INSTALACIÓN DE DEPENDENCIAS NVIDIA/CUDA PARA EC2 (GPU)
# =============================================================================

# Archivo de estado para tracking de instalación
NVIDIA_INSTALL_STATE_FILE="/opt/subs-gp/nvidia_install_state"
NVIDIA_REBOOT_COUNT_FILE="/opt/subs-gp/nvidia_reboot_count"
MAX_AUTO_REBOOTS=2  # Máximo 2 reboots automáticos antes de error fatal

# =============================================================================
# FUNCIONES DE MANEJO DE ESTADO Y REBOOTS
# =============================================================================

get_reboot_count() {
    if [ -f "$NVIDIA_REBOOT_COUNT_FILE" ]; then
        local stored
        stored=$(cat "$NVIDIA_REBOOT_COUNT_FILE" 2>/dev/null || echo "0:unknown")
        local stored_count stored_kernel
        stored_count=$(echo "$stored" | cut -d: -f1)
        stored_kernel=$(echo "$stored" | cut -d: -f2-)
        # Si el kernel cambió (ej: update del SO), resetear el contador automáticamente
        if [ "$stored_kernel" != "$(uname -r)" ]; then
            echo "ℹ️  Kernel cambió de '$stored_kernel' a '$(uname -r)'. Reseteando contador de reboots." >&2
            sudo rm -f "$NVIDIA_REBOOT_COUNT_FILE" "$NVIDIA_INSTALL_STATE_FILE"
            echo "0"
        else
            echo "$stored_count"
        fi
    else
        echo "0"
    fi
}

increment_reboot_count() {
    local current=$(get_reboot_count)
    local new_count=$((current + 1))
    sudo mkdir -p "$(dirname "$NVIDIA_REBOOT_COUNT_FILE")"
    # Guardar "count:kernel" para detectar cambios de kernel
    echo "${new_count}:$(uname -r)" | sudo tee "$NVIDIA_REBOOT_COUNT_FILE" > /dev/null
    echo "$new_count"
}

reset_reboot_count() {
    sudo rm -f "$NVIDIA_REBOOT_COUNT_FILE"
    sudo rm -f "$NVIDIA_INSTALL_STATE_FILE"
}

diagnose_nvidia_failure() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🔍 DIAGNÓSTICO DE FALLA NVIDIA                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    echo "📋 Kernel actual: $(uname -r)"
    echo ""
    
    echo "📋 GPU NVIDIA en hardware (lspci):"
    lspci 2>/dev/null | grep -i nvidia || echo "   ⚠️  (ninguno — posiblemente no hay GPU o lspci no disponible)"
    echo ""

    echo "📋 Módulos NVIDIA cargados:"
    lsmod | grep nvidia || echo "   (ninguno)"
    echo ""
    
    echo "📋 Device nodes /dev/nvidia*:"
    ls -la /dev/nvidia* 2>/dev/null || echo "   (ninguno — módulo no cargado)"
    echo ""
    
    echo "📋 Estado DKMS:"
    dkms status 2>/dev/null | grep nvidia || echo "   (ninguno)"
    echo ""

    echo "📋 Headers de kernel disponibles (/usr/src/kernels/):"
    ls /usr/src/kernels/ 2>/dev/null || echo "   (ninguno)"
    echo ""
    
    echo "📋 Últimos mensajes de nvidia en dmesg:"
    dmesg 2>/dev/null | grep -i nvidia | tail -10 || echo "   (ninguno)"
    echo ""
    
    echo "📋 Paquetes NVIDIA instalados:"
    if [ "$RHEL_MODE" = "true" ]; then
        rpm -qa | grep -i nvidia | head -10 || echo "   (ninguno)"
    else
        dpkg -l | grep -i nvidia | head -10 || echo "   (ninguno)"
    fi
    echo ""

    if [ -f /tmp/dkms-build.log ] && [ -s /tmp/dkms-build.log ]; then
        echo "📋 Último log de compilación DKMS (/tmp/dkms-build.log):"
        tail -30 /tmp/dkms-build.log
        echo ""
    fi
}

ask_reboot() {
    local stage=$1
    local reboot_count=$(get_reboot_count)
    local current_kernel
    current_kernel=$(uname -r)
    
    # Verificar si hemos excedido el límite de reboots
    if [ "$reboot_count" -ge "$MAX_AUTO_REBOOTS" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║  🛑 ERROR FATAL: Límite de reboots automáticos alcanzado             ║"
        echo "║                                                                      ║"
        echo "║  Se han realizado $reboot_count reboots sin éxito.                       ║"
        echo "║  El driver NVIDIA no logra comunicarse con el kernel.                ║"
        echo "║                                                                      ║"
        echo "║  Posibles causas:                                                    ║"
        echo "║  - kernel-devel no disponible para el kernel personalizado           ║"
        echo "║    (causa más común en RHEL modificado/AWS)                          ║"
        echo "║  - Incompatibilidad de versión: driver vs kernel                     ║"
        echo "║  - GPU no soportada por el driver                                    ║"
        echo "║  - Problema de hardware                                              ║"
        echo "║                                                                      ║"
        echo "║  Acciones recomendadas:                                              ║"
        echo "║  1. Verificar GPU:  lspci | grep -i nvidia                           ║"
        echo "║  2. Ver log DKMS:   cat /tmp/dkms-build.log                          ║"
        echo "║  3. Revisar dmesg:  dmesg | grep -i nvidia                           ║"
        echo "║  4. Verificar headers:                                               ║"
        echo "║       ls /usr/src/kernels/                                           ║"
        echo "║       sudo dnf list kernel-devel                                     ║"
        echo "║  5. Forzar DKMS manualmente:                                         ║"
        printf "║       sudo dkms install nvidia/%s -k %s\n" \
            "$(dkms status 2>/dev/null | grep -oP 'nvidia/\K[0-9.]+' | head -1)" \
            "$current_kernel" | head -c 72
        echo "  ║"
        echo "║  6. Revisar logs: journalctl -u ire-ei -n 100                        ║"
        echo "║                                                                      ║"
        echo "║  Para reiniciar el contador de intentos:                             ║"
        echo "║  sudo rm -f $NVIDIA_REBOOT_COUNT_FILE                 ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        echo ""
        
        diagnose_nvidia_failure
        
        # No borrar archivos de estado para poder debuggear
        exit 2  # Código especial para error fatal
    fi
    
    # Incrementar contador y guardar estado
    local new_count=$(increment_reboot_count)
    sudo mkdir -p "$(dirname "$NVIDIA_INSTALL_STATE_FILE")"
    echo "$stage" | sudo tee "$NVIDIA_INSTALL_STATE_FILE" > /dev/null
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🔄 REINICIO AUTOMÁTICO ($new_count/$MAX_AUTO_REBOOTS)                              ║"
    echo "║                                                                      ║"
    echo "║  Los drivers NVIDIA requieren un reinicio del sistema.               ║"
    echo "║  El sistema se reiniciará automáticamente en 5 segundos...           ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    sleep 5
    echo "🔄 Reiniciando el sistema..."
    sudo reboot
    exit 0
}

# =============================================================================
# FUNCIÓN: Instalar kernel-devel con múltiples estrategias de fallback
# Necesario para que DKMS pueda compilar el módulo NVIDIA
# =============================================================================
try_install_kernel_devel() {
    local current_kernel="$1"
    local DKMS_LOG="${2:-/tmp/dkms-build.log}"

    echo "   [DKMS] Verificando kernel-devel para kernel $current_kernel..."

    # Si ya están los headers instalados, salir rápido
    if [ -d "/usr/src/kernels/${current_kernel}" ]; then
        echo "   [DKMS] Headers ya disponibles en /usr/src/kernels/${current_kernel}"
        return 0
    fi

    # Estrategia 1: Paquete exacto para el kernel en ejecución
    echo "   [DKMS] Estrategia 1: kernel-devel-${current_kernel} (match exacto)..."
    if sudo dnf install -y "kernel-devel-${current_kernel}" 2>&1 | tee -a "$DKMS_LOG"; then
        if [ -d "/usr/src/kernels/${current_kernel}" ]; then
            echo "   [DKMS] ✅ kernel-devel instalado (estrategia 1)"
            return 0
        fi
    fi

    # Estrategia 2: Habilitar repos BaseOS/AppStream explícitamente (común en RHEL AWS modificado)
    echo "   [DKMS] Estrategia 2: kernel-devel habilitando repos BaseOS/AppStream..."
    if sudo dnf install -y --enablerepo="baseos,appstream" "kernel-devel-${current_kernel}" 2>&1 | tee -a "$DKMS_LOG"; then
        if [ -d "/usr/src/kernels/${current_kernel}" ]; then
            echo "   [DKMS] ✅ kernel-devel instalado (estrategia 2)"
            return 0
        fi
    fi

    # Estrategia 3: kernel-devel genérico (versión más reciente disponible)
    echo "   [DKMS] Estrategia 3: kernel-devel genérico (puede no coincidir exactamente)..."
    if sudo dnf install -y kernel-devel kernel-headers 2>&1 | tee -a "$DKMS_LOG"; then
        # Verificar si hay alguna versión instalada (aunque no sea exacta)
        if ls /usr/src/kernels/ 2>/dev/null | head -1 | grep -q .; then
            local avail_kernel
            avail_kernel=$(ls /usr/src/kernels/ | tail -1)
            echo "   [DKMS] ⚠️  kernel-devel instalado para $avail_kernel (running: $current_kernel)"
            echo "   [DKMS]    Puede funcionar si las versiones son compatibles."
            return 0
        fi
    fi

    # Estrategia 4: Buscar en todos los repos disponibles
    echo "   [DKMS] Estrategia 4: búsqueda en todos los repos habilitados..."
    local found_pkg
    found_pkg=$(sudo dnf search "kernel-devel-${current_kernel}" 2>/dev/null | grep -i "kernel-devel" | head -1 | awk '{print $1}')
    if [ -n "$found_pkg" ]; then
        echo "   [DKMS] Encontrado: $found_pkg. Instalando..."
        sudo dnf install -y "$found_pkg" 2>&1 | tee -a "$DKMS_LOG" || true
        if [ -d "/usr/src/kernels/${current_kernel}" ]; then
            echo "   [DKMS] ✅ kernel-devel instalado (estrategia 4)"
            return 0
        fi
    fi

    echo "   [DKMS] ❌ No se pudo instalar kernel-devel para $current_kernel."
    echo "   [DKMS]    Repos disponibles:"
    sudo dnf repolist 2>/dev/null | tee -a "$DKMS_LOG" || true
    echo "   [DKMS]    Headers disponibles en /usr/src/kernels/:"
    ls /usr/src/kernels/ 2>/dev/null || echo "   [DKMS]    (ninguno)"
    return 1
}

# Función para compilar e instalar módulo NVIDIA con DKMS
# Retorna 0 si el módulo quedó funcional, 1 si falló
try_dkms_build() {
    local current_kernel="$1"
    local DKMS_LOG="${2:-/tmp/dkms-build.log}"

    if ! command -v dkms &>/dev/null; then
        echo "   [DKMS] dkms no está instalado, saltando compilación."
        return 1
    fi

    local nvidia_dkms_ver
    nvidia_dkms_ver=$(dkms status 2>/dev/null | grep -oP 'nvidia/\K[0-9.]+' | head -1)
    if [ -z "$nvidia_dkms_ver" ]; then
        echo "   [DKMS] No hay módulo nvidia registrado en DKMS."
        return 1
    fi

    echo "   [DKMS] Módulo DKMS detectado: nvidia/$nvidia_dkms_ver"
    echo "   [DKMS] Kernel objetivo: $current_kernel"
    echo "" >> "$DKMS_LOG"
    echo "=== dkms build/install $(date) ===" >> "$DKMS_LOG"

    # Asegurar que el módulo esté en estado correcto (remove si estaba en bad state)
    dkms status 2>/dev/null | grep "nvidia/${nvidia_dkms_ver}.*${current_kernel}" >> "$DKMS_LOG" 2>&1 || true

    echo "   [DKMS] Ejecutando: dkms install nvidia/$nvidia_dkms_ver -k $current_kernel..."
    local dkms_rc=0
    sudo dkms install "nvidia/$nvidia_dkms_ver" -k "$current_kernel" \
        --force 2>&1 | tee -a "$DKMS_LOG" || dkms_rc=$?

    # Verificar si realmente quedó instalado
    if dkms status 2>/dev/null | grep -q "nvidia/${nvidia_dkms_ver}.*${current_kernel}.*installed"; then
        echo "   [DKMS] ✅ Módulo instalado correctamente."
        return 0
    fi

    # Segundo intento con autoinstall
    echo "   [DKMS] Primera instalación no confirmada. Intentando dkms autoinstall..."
    sudo dkms autoinstall -k "$current_kernel" 2>&1 | tee -a "$DKMS_LOG" || true

    if dkms status 2>/dev/null | grep -q "nvidia/${nvidia_dkms_ver}.*${current_kernel}.*installed"; then
        echo "   [DKMS] ✅ Módulo instalado via autoinstall."
        return 0
    fi

    echo "   [DKMS] ❌ DKMS build falló. Últimas líneas del log ($DKMS_LOG):"
    tail -20 "$DKMS_LOG"
    echo "   [DKMS] Estado DKMS actual:"
    dkms status 2>/dev/null || true
    return 1
}

# Función para asegurar que los módulos NVIDIA estén cargados y listos
# Esto resuelve el problema de que nvidia-smi no responda después del primer reinicio
ensure_nvidia_ready() {
    local max_attempts=${1:-10}  # Número de intentos (default 10)
    local wait_time=${2:-3}      # Segundos entre intentos (default 3)
    local attempt=0
    local DKMS_LOG="/tmp/dkms-build.log"
    
    echo "🔧 Inicializando módulos NVIDIA (máx $max_attempts intentos)..."

    # Compilar módulo NVIDIA si DKMS no lo tiene para el kernel actual
    if command -v dkms &>/dev/null; then
        if ! modinfo -F vermagic nvidia 2>/dev/null | grep -q "$(uname -r)"; then
            local current_kernel
            current_kernel=$(uname -r)
            echo "   [DKMS] Módulo nvidia no compilado para $current_kernel."

            # Intentar obtener kernel-devel con múltiples estrategias
            try_install_kernel_devel "$current_kernel" "$DKMS_LOG" || true

            # Intentar compilar el módulo
            if try_dkms_build "$current_kernel" "$DKMS_LOG"; then
                echo "   [DKMS] Módulo compilado. Cargando..."
            else
                echo "   [DKMS] Compilación falló. Se intentará cargar módulo preexistente si hay alguno."
            fi
        fi
    fi

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        
        # Cargar todos los módulos NVIDIA necesarios
        sudo modprobe nvidia 2>/dev/null || true
        sudo modprobe nvidia_uvm 2>/dev/null || true
        sudo modprobe nvidia_drm 2>/dev/null || true

        # Crear device nodes si no existen (necesario cuando udev no ha procesado las reglas)
        if ! ls /dev/nvidia* &>/dev/null; then
            if command -v nvidia-modprobe &>/dev/null; then
                sudo nvidia-modprobe -u -c=0 2>/dev/null || true
            else
                sudo udevadm trigger --action=add --subsystem-match=module 2>/dev/null || true
                sudo udevadm settle --timeout=5 2>/dev/null || true
            fi
        fi

        # Iniciar nvidia-persistenced si existe
        if systemctl list-unit-files 2>/dev/null | grep -q nvidia-persistenced; then
            sudo systemctl start nvidia-persistenced 2>/dev/null || true
        fi
        
        # Esperar un momento
        sleep $wait_time
        
        # Verificar si nvidia-smi responde
        if nvidia-smi &>/dev/null; then
            echo "✅ Driver NVIDIA listo (intento $attempt/$max_attempts)"
            return 0
        fi

        # Diagnóstico en cada intento fallido
        if [ $attempt -eq 1 ] || [ $attempt -eq $max_attempts ]; then
            echo "   [DIAG intento $attempt] /dev/nvidia*:"
            ls -la /dev/nvidia* 2>&1 || echo "   [DIAG] No existen /dev/nvidia*"
            echo "   [DIAG] nvidia-smi output:"
            nvidia-smi 2>&1 | head -5 || true
            echo "   [DIAG] dmesg nvidia (últimas 5 líneas):"
            dmesg 2>/dev/null | grep -i nvidia | tail -5 || true
        fi

        if [ $attempt -lt $max_attempts ]; then
            echo "   Esperando driver NVIDIA... (intento $attempt/$max_attempts)"
        fi
    done
    
    echo "⚠️  Driver NVIDIA no respondió después de $max_attempts intentos"
    return 1
}

setup_rhel_nvidia_repo() {
    echo "📦 Configurando repositorio NVIDIA para RHEL 9..."
    
    # Instalar EPEL si no está instalado (para RHEL 9 en AWS)
    if ! rpm -q epel-release &> /dev/null; then
        echo "📦 Instalando EPEL para RHEL 9..."
        # En RHEL 9 de AWS, EPEL no está en los repos base, usar URL directa
        sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || {
            echo "⚠️  No se pudo instalar EPEL, continuando sin él..."
        }
    fi
    
    # Agregar repositorio CUDA de NVIDIA para RHEL 9
    if [ ! -f /etc/yum.repos.d/cuda-rhel9.repo ]; then
        echo "📦 Agregando repositorio CUDA de NVIDIA..."
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo || {
            echo "⚠️  No se pudo agregar el repo de CUDA, intentando método alternativo..."
            # Método alternativo: descargar y configurar manualmente
            sudo tee /etc/yum.repos.d/cuda-rhel9.repo << 'REPO'
[cuda-rhel9-x86_64]
name=cuda-rhel9-x86_64
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64
enabled=1
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/D42D0685.pub
REPO
        }
    fi
    
    # Limpiar caché de dnf
    sudo dnf clean all
    sudo dnf makecache || true
    
    echo "✅ Repositorio NVIDIA configurado"
}

setup_ubuntu_nvidia_repo() {
    echo "📦 Configurando repositorio NVIDIA para Ubuntu..."
    
    # Asegurar que wget esté instalado
    sudo apt-get update -y
    sudo apt-get install -y wget

    # Detectar versión de Ubuntu
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        
        # VERSION_ID suele ser "22.04" o "20.04"
        # Quitamos el punto para obtener "2204" o "2004"
        UBUNTU_VER_NO_DOT=$(echo "$VERSION_ID" | tr -d '.')
        
        echo "ℹ️  Detectado Ubuntu $VERSION_ID"
        
        # URL del keyring
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER_NO_DOT}/x86_64/cuda-keyring_1.1-1_all.deb"
        KEYRING_FILE="cuda-keyring_1.1-1_all.deb"
        
        echo "⬇️  Descargando keyring de NVIDIA: $KEYRING_URL"
        if wget -q "$KEYRING_URL"; then
            echo "📦 Instalando keyring..."
            sudo dpkg -i "$KEYRING_FILE"
            rm "$KEYRING_FILE"
            
            echo "🔄 Actualizando repositorios..."
            sudo apt-get update
        else
            echo "⚠️  No se pudo descargar el keyring para ubuntu${UBUNTU_VER_NO_DOT}. Intentando continuar..."
        fi
    fi
}

install_nvidia_dependencies() {
    echo "🎮 Verificando dependencias NVIDIA/CUDA..."
    
    local needs_reboot=false
    local reboot_count=$(get_reboot_count)
    
    # Mostrar estado actual si hay intentos previos
    if [ "$reboot_count" -gt 0 ]; then
        echo "ℹ️  Intento de instalación #$((reboot_count + 1)) (máx $MAX_AUTO_REBOOTS reboots automáticos)"
    fi
    
    # Configurar repositorios
    if [ "$RHEL_MODE" = "true" ]; then
        setup_rhel_nvidia_repo
    else
        setup_ubuntu_nvidia_repo
    fi
    
    # ETAPA 1: Instalar driver NVIDIA
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  nvidia-smi no encontrado. Instalando drivers NVIDIA..."
        
        pkg_update
        
        if [ "$RHEL_MODE" = "true" ]; then
            # RHEL: Usar SIEMPRE driver precompilado (kmod) - más confiable en AWS
            # Los kernels de AWS son personalizados y DKMS frecuentemente falla
            echo "📦 Instalando NVIDIA Driver precompilado para RHEL 9..."
            
            # Desactivar set -e temporalmente para manejar errores de dnf
            set +e
            
            # Log para debugging
            echo "$(date): Iniciando instalación de nvidia-driver" | sudo tee -a "$INSTALL_LOG"
            
            # Primero resetear el módulo si existe para evitar conflictos
            echo "   Reseteando módulo nvidia-driver..."
            sudo dnf module reset nvidia-driver -y 2>&1 | sudo tee -a "$INSTALL_LOG"
            
            # Instalar el driver
            echo "   Ejecutando: dnf module install nvidia-driver:latest"
            install_output=$(sudo dnf module install -y nvidia-driver:latest 2>&1)
            install_rc=$?
            echo "$install_output" | sudo tee -a "$INSTALL_LOG"
            
            if [ $install_rc -ne 0 ]; then
                echo "⚠️  Falló instalación con módulo (código: $install_rc)"
                echo "   Salida: $install_output"
                echo "   Intentando paquete directo..."
                
                install_output=$(sudo dnf install -y nvidia-driver 2>&1)
                install_rc=$?
                echo "$install_output" | sudo tee -a "$INSTALL_LOG"
                
                if [ $install_rc -ne 0 ]; then
                    echo "❌ Error instalando driver NVIDIA (código: $install_rc)"
                    echo "   Limpiando caché y reintentando..."
                    sudo dnf clean all
                    sudo dnf makecache
                    
                    # Último intento
                    install_output=$(sudo dnf module install -y nvidia-driver:latest 2>&1)
                    install_rc=$?
                    echo "$install_output" | sudo tee -a "$INSTALL_LOG"
                    
                    if [ $install_rc -ne 0 ]; then
                        echo "❌ No se pudo instalar el driver NVIDIA después de varios intentos"
                        echo "   Reiniciando para intentar nuevamente..."
                        set -e  # Reactivar
                        ask_reboot "driver_install_failed"
                    fi
                fi
            fi
            
            echo "$(date): Instalación de nvidia-driver completada" | sudo tee -a "$INSTALL_LOG"

            # Instalar kernel-devel con multi-fallback y compilar módulo DKMS
            RUNNING_KERNEL=$(uname -r)
            try_install_kernel_devel "$RUNNING_KERNEL" "$INSTALL_LOG" || true
            try_dkms_build "$RUNNING_KERNEL" "$INSTALL_LOG" && {
                sudo modprobe nvidia 2>/dev/null || true
                sudo modprobe nvidia_uvm 2>/dev/null || true
                sudo modprobe nvidia_drm 2>/dev/null || true
            } || echo "⚠️  DKMS build inicial falló. Se intentará de nuevo post-reboot."

            # Reactivar set -e
            set -e
        else
            # Ubuntu: Instalar driver NVIDIA
            echo "📦 Instalando NVIDIA Driver 580..."
            sudo apt-get install -y nvidia-driver-580
        fi
        
        echo "✅ Driver NVIDIA instalado."
        needs_reboot=true
    else
        # El comando nvidia-smi existe, intentar inicializar los módulos con múltiples intentos
        if ensure_nvidia_ready 15 3; then
            # Driver funcionando correctamente
            :
        else
            echo "⚠️  nvidia-smi existe pero no responde después de inicializar módulos"
            RUNNING_KERNEL=$(uname -r)
            DKMS_STATUS=$(dkms status 2>/dev/null | grep "^nvidia" | head -1)
            echo "   DKMS status: ${DKMS_STATUS:-ninguno}"

            # Intentar DKMS con las funciones mejoradas (multi-fallback kernel-devel)
            if echo "$DKMS_STATUS" | grep -qE ': added|: built'; then
                echo "📦 Módulo DKMS no instalado para kernel $RUNNING_KERNEL. Compilando con multi-fallback..."
                try_install_kernel_devel "$RUNNING_KERNEL" "$INSTALL_LOG" || true
                if try_dkms_build "$RUNNING_KERNEL" "$INSTALL_LOG"; then
                    echo "✅ Módulo compilado vía DKMS. Cargando..."
                    sudo modprobe nvidia 2>/dev/null || true
                    sudo modprobe nvidia_uvm 2>/dev/null || true
                    sudo modprobe nvidia_drm 2>/dev/null || true
                else
                    # DKMS falló definitivamente → intentar kmod precompilado como último recurso
                    echo "⚠️  DKMS falló. Intentando instalar kmod precompilado de NVIDIA como alternativa..."
                    set +e
                    # El stream "kmod" de nvidia-driver usa módulos precompilados (sin DKMS)
                    sudo dnf module reset nvidia-driver -y 2>&1 | sudo tee -a "$INSTALL_LOG" || true
                    if sudo dnf module install -y nvidia-driver:latest/kmod 2>&1 | sudo tee -a "$INSTALL_LOG"; then
                        echo "✅ kmod precompilado instalado. Cargando módulos..."
                        sudo modprobe nvidia 2>/dev/null || true
                        sudo modprobe nvidia_uvm 2>/dev/null || true
                        sudo modprobe nvidia_drm 2>/dev/null || true
                    else
                        echo "⚠️  kmod precompilado tampoco disponible para este kernel."
                        echo "   Causa probable: kernel personalizado sin kmod precompilado correspondiente."
                        echo "   Se requiere kernel-devel del kernel $RUNNING_KERNEL para compilar DKMS."
                    fi
                    set -e
                fi
            fi
        fi
    fi

    # Verificar si el driver funciona después de todos los intentos
    if ! nvidia-smi &> /dev/null; then
        echo "❌ El driver NVIDIA está instalado pero no comunica con el kernel."
        echo "   Solicitando reboot automático..."
        ask_reboot "driver_installed"
    else
        echo "✅ Driver NVIDIA ya está instalado y funcionando"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        
        # Driver funciona - resetear contadores de estado
        reset_reboot_count
        
        # Habilitar persistencia para futuros arranques
        sudo systemctl enable nvidia-persistenced &> /dev/null || true
    fi
    
    # ETAPA 2: Instalar CUDA Toolkit
    
    # Intentar cargar CUDA al PATH si existe pero no está cargado
    # Intentar cargar CUDA al PATH si existe pero no está cargado
    if ! command -v nvcc &> /dev/null; then
        if [ -x "/usr/local/cuda/bin/nvcc" ]; then
            echo "⚠️  nvcc detectado en /usr/local/cuda/bin/nvcc. Agregando al PATH..."
            export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
            export CUDA_HOME=/usr/local/cuda
        elif [ -d "/usr/local/cuda/bin" ]; then
            echo "⚠️  CUDA detectado en /usr/local/cuda pero no en PATH. Agregando..."
            export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
            export CUDA_HOME=/usr/local/cuda
        fi
    fi

    # Verificar nuevamente (modificado para aceptar ruta absoluta como éxito)
    if ! command -v nvcc &> /dev/null && [ ! -x "/usr/local/cuda/bin/nvcc" ]; then

        echo "📦 Instalando CUDA Toolkit..."
        pkg_update
        
        if [ "$RHEL_MODE" = "true" ]; then
            # RHEL: CUDA toolkit
            sudo dnf install -y cuda-toolkit-12-4
        else
            # Ubuntu
            # Instalar toolkit con fallback: 12.4 -> 12.5 -> 12 (genérico)
            if ! sudo apt-get install -y cuda-toolkit-12-4; then
                echo "⚠️  No se encontró cuda-toolkit-12-4, intentando con 12-5..."
                if ! sudo apt-get install -y cuda-toolkit-12-5; then
                    echo "⚠️  No se encontró cuda-toolkit-12-5, intentando con cuda-toolkit-12..."
                    if ! sudo apt-get install -y cuda-toolkit-12; then
                        echo "❌ Error: No se pudo instalar ninguna versión de CUDA Toolkit."
                        exit 1
                    fi
                fi
            fi
        fi
        
        # Agregar CUDA al PATH permanentemente
        local shell_rc="$HOME/.bashrc"
        if [ "$RHEL_MODE" = "true" ] && [ -f "$HOME/.bash_profile" ]; then
            shell_rc="$HOME/.bash_profile"
        fi
        
        if ! grep -q "/usr/local/cuda/bin" "$shell_rc"; then
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$shell_rc"
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "$shell_rc"
        fi
        
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        
        # Asegurar exportación de variables explícitamente tras instalación
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        export CUDA_HOME=/usr/local/cuda
        
        echo "✅ CUDA Toolkit instalado"
        needs_reboot=true
    else
        echo "✅ CUDA ya está instalado: $(nvcc --version | grep release)"
        # Validar que las variables de entorno sean correctas incluso si ya estaba instalado
        if [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
             echo "⚠️  Corrigiendo PATH para incluir CUDA..."
             export PATH=/usr/local/cuda/bin:$PATH
        fi
        if [[ ":$LD_LIBRARY_PATH:" != *":/usr/local/cuda/lib64:"* ]]; then
             echo "⚠️  Corrigiendo LD_LIBRARY_PATH para incluir CUDA..."
             export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        fi
    fi
    
    # Validación explícita requerida por usuario
    if ! echo "$PATH" | grep -q "cuda"; then
        echo "⚠️  Advertencia: Parece que CUDA no está en el PATH correctamente."
    else
        echo "✅ Validación de entorno CUDA: OK"
    fi

    
    # Si se instaló driver o CUDA, requiere reinicio antes de TensorRT
    if [ "$needs_reboot" = true ]; then
        echo "🧹 Limpiando caché de paquetes..."
        pkg_clean
        
        ask_reboot "nvidia_cuda_installed"
    fi
    
    # ETAPA 3: Instalar TensorRT (después del reinicio)
    # Para RHEL 9, usar el meta-paquete "tensorrt" del repo de NVIDIA
    if ! pkg_check_installed "tensorrt"; then
        echo "📦 Instalando TensorRT y dependencias de inferencia..."
        pkg_update
        
        if [ "$RHEL_MODE" = "true" ]; then
            # En RHEL 9, usar el meta-paquete tensorrt
            # En RHEL 9, intentar varios nombres de paquetes
            if ! sudo dnf install -y tensorrt 2>&1; then
                echo "⚠️  Meta-paquete 'tensorrt' no encontrado, intentando paquetes individuales modernos..."
                
                # Intentar paquetes modernos (TRT 10? o estructura libs/devel)
                if ! sudo dnf install -y tensorrt-libs tensorrt-devel 2>&1; then
                     echo "⚠️  Paquetes 'tensorrt-libs' no encontrados, intentando estilo legacy (libnvinfer)..."
                     
                     # Intentar estilo antiguo/legacy (TRT 8 para CUDA 11/12)
                     # Nota: libnvinfer8 suele ser para TRT 8.x
                     if ! sudo dnf install -y libnvinfer8 libnvinfer-plugin8 libnvinfer-devel libnvinfer-plugin-devel 2>&1; then
                         echo "❌ Error: No se pudo instalar TensorRT de ninguna forma conocida."
                         echo "   Verifica que el repositorio NVIDIA CUDA esté configurado correctamente."
                         exit 1
                     fi
                fi
            fi
            
            # Instalar bindings de Python si existen (opcional pero recomendado)
            sudo dnf install -y python3-libnvinfer 2>/dev/null || true
        else
            sudo apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev
        fi
        echo "✅ TensorRT instalado"
    else
        echo "✅ TensorRT ya está instalado"
    fi
    
    # ETAPA 4: Instalar cuDNN
    local cudnn_pkg="libcudnn9"
    if [ "$RHEL_MODE" = "true" ]; then
        cudnn_pkg="libcudnn9-cuda-12"
    fi
    
    if ! pkg_check_installed "$cudnn_pkg"; then
        echo "📦 Instalando cuDNN 9 para CUDA 12..."
        if [ "$RHEL_MODE" = "true" ]; then
            if ! sudo dnf install -y libcudnn9-cuda-12 libcudnn9-devel-cuda-12 2>&1; then
                # Intentar versión alternativa
                if ! sudo dnf install -y cudnn 2>&1; then
                    echo "❌ Error: No se pudo instalar cuDNN. Es una dependencia crítica."
                    exit 1
                fi
            fi
        else
            sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
        fi
        echo "✅ cuDNN instalado"
    else
        echo "✅ cuDNN ya está instalado"
    fi
    
    # Limpiar caché
    echo "🧹 Limpiando caché de paquetes..."
    pkg_clean
    
    echo "✅ Todas las dependencias NVIDIA/CUDA verificadas"
}

# =============================================================================
# INSTALACIÓN DE PYTHON Y DEPENDENCIAS
# =============================================================================

PYENV_ROOT="$HOME/.pyenv"

# Verificar si ya existe una instalación completa
if [ -f "$FULL_INSTALL_STATE_FILE" ]; then
    echo "⚡ Instalación completa detectada ($FULL_INSTALL_STATE_FILE)."
    echo "🔍 Verificando componentes críticos..."
    
    # Cargar entorno mínimo necesario
    export PYENV_ROOT
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init -)" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
    
    # -------------------------------------------------------------------------
    # VERIFICACIONES CRÍTICAS (incluso con marker de instalación completa)
    # -------------------------------------------------------------------------
    
    # 1. Verificar NVIDIA/GPU (si no se omitió)
    if [[ "$SKIP_NVIDIA" == false ]] && [[ "$(uname)" == "Linux" ]]; then
        if lspci 2>/dev/null | grep -i nvidia &> /dev/null; then
            if ! nvidia-smi &> /dev/null; then
                echo "⚠️  nvidia-smi no responde. Intentando cargar módulos..."
                if ! ensure_nvidia_ready 10 3; then
                    echo "❌ GPU no disponible. Borrando marker para reinstalar..."
                    sudo rm -f "$FULL_INSTALL_STATE_FILE"
                    exec "$0" "$@"  # Reiniciar script
                fi
            fi
            echo "✅ GPU verificada: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'OK')"
        fi
    fi
    
    # 2. Verificar Python
    if ! pyenv versions --bare 2>/dev/null | grep -q "^${PYTHON_VERSION}$"; then
        echo "⚠️  Python $PYTHON_VERSION no encontrado en pyenv. Instalando..."
        CPPFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib64" pyenv install "$PYTHON_VERSION" -s
        echo "✅ Python $PYTHON_VERSION instalado"
    else
        echo "✅ Python $PYTHON_VERSION verificado"
    fi
    
    # 3. Verificar Poetry (se verificará/instalará más adelante con pyenv)
    #    Aquí solo notificamos que será configurado después
    echo "✅ Poetry será verificado con pyenv Python $PYTHON_VERSION"
    
else
    # -------------------------------------------------------------------------
    # BLOQUE DE INSTALACIÓN (Se ejecuta solo si no existe el marker)
    # -------------------------------------------------------------------------

    # Detectar si estamos en Linux y si hay GPU disponible
    if [[ "$(uname)" == "Linux" ]] && [[ "$SKIP_NVIDIA" == false ]]; then
        # Verificar si hay GPU NVIDIA en el sistema
        if lspci 2>/dev/null | grep -i nvidia &> /dev/null || [ -f /proc/driver/nvidia/version ]; then
            install_nvidia_dependencies
        else
            echo "ℹ️  No se detectó GPU NVIDIA, omitiendo instalación de CUDA/TensorRT"
        fi
    elif [[ "$SKIP_NVIDIA" == true ]]; then
        echo "ℹ️  Instalación NVIDIA omitida por --no-nvidia"
    fi

    echo "🔍 Verificando dependencias de Python..."

    # Verificar si las dependencias de compilación están instaladas
    check_build_deps_installed() {
        if [ "$RHEL_MODE" = "true" ]; then
            # Verificar paquetes esenciales en RHEL
            rpm -q gcc make openssl-devel zlib-devel libffi-devel git curl &> /dev/null
        else
            # Verificar paquetes esenciales en Ubuntu
            dpkg -l | grep -q build-essential && dpkg -l | grep -q libssl-dev
        fi
    }

    # Instalar dependencias de compilación para pyenv (necesarias para compilar Python)
    install_pyenv_build_deps() {
        # Si ya están instaladas, no hacer nada
        if check_build_deps_installed; then
            echo "✅ Dependencias de compilación ya están instaladas"
            return 0
        fi
        
        echo "📦 Instalando dependencias de compilación para Python..."
        
        # Desactivar temporalmente set -e para manejar errores
        set +e
        
        if [ "$RHEL_MODE" = "true" ]; then
            # RHEL 9: Dependencias para compilar Python
            echo "📦 Instalando grupo Development Tools..."
            sudo dnf groupinstall -y "Development Tools" 2>&1 || {
                echo "⚠️  No se pudo instalar Development Tools como grupo, instalando paquetes individuales..."
                sudo dnf install -y gcc gcc-c++ make automake autoconf 2>&1
            }
            
            echo "📦 Instalando librerías de desarrollo..."
            sudo dnf install -y \
                git curl \
                zlib-devel bzip2-devel openssl-devel ncurses-devel \
                sqlite-devel readline-devel tk-devel \
                libffi-devel xz-devel libuuid-devel 2>&1 || {
                echo "⚠️  Algunos paquetes no se pudieron instalar, continuando..."
            }
        else
            # Ubuntu/Debian
            sudo apt-get update -y
            sudo apt-get install -y \
                build-essential libssl-dev zlib1g-dev libbz2-dev \
                libreadline-dev libsqlite3-dev wget curl llvm \
                libncursesw5-dev xz-utils tk-dev libxml2-dev \
                libxmlsec1-dev libffi-dev liblzma-dev 2>&1 || {
                echo "⚠️  Algunos paquetes no se pudieron instalar, continuando..."
            }
        fi
        
        # Reactivar set -e
        set -e
        
        echo "✅ Dependencias de compilación procesadas"
    }

    # Configurar pyenv si existe el directorio
    if [ -d "$PYENV_ROOT" ]; then
        echo "✅ pyenv detectado en $PYENV_ROOT"
        export PYENV_ROOT
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
        eval "$(pyenv init -)" 2>/dev/null || true
    else
        echo "⚠️  pyenv no está instalado. Instalando..."
        
        # Instalar dependencias de compilación ANTES de pyenv
        install_pyenv_build_deps
        
        curl https://pyenv.run | bash
        
        export PYENV_ROOT
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
        eval "$(pyenv init -)"
        
        echo "✅ pyenv instalado correctamente"
    fi

    # Validar que PYTHON_VERSION no esté vacía
    if [ -z "$PYTHON_VERSION" ]; then
        echo "❌ Error: No se pudo leer la versión de Python desde .python-version"
        exit 1
    fi

    # Instalar la versión de Python con pyenv
    echo "🐍 Verificando Python $PYTHON_VERSION..."
    if pyenv versions --bare 2>/dev/null | grep -q "^${PYTHON_VERSION}$"; then
        echo "✅ Python $PYTHON_VERSION ya está instalado"
    else
        echo "⏳ Instalando Python $PYTHON_VERSION (esto puede tardar)..."
        CPPFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib64" pyenv install "$PYTHON_VERSION" -s
        echo "✅ Python $PYTHON_VERSION instalado correctamente"
    fi

    # Marcar instalación base como completa
    echo "✅ Instalación base completada."
    sudo touch "$FULL_INSTALL_STATE_FILE"
    sudo chmod 666 "$FULL_INSTALL_STATE_FILE"
fi

# Usar la versión de Python localmente
echo "📌 Estableciendo Python local..."
pyenv local "$PYTHON_VERSION" 2>/dev/null || true
pyenv rehash

# Forzar la versión de Python en el entorno
export PYENV_VERSION="$PYTHON_VERSION"

# Obtener el path absoluto al Python de pyenv (NO usar shims)
PYENV_PYTHON_PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python"
PYENV_PIP_PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/pip"
PYENV_POETRY_PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/poetry"

echo "   Verificando Python en: $PYENV_PYTHON_PATH"

# Verificar que el binario de Python existe y funciona
if [ ! -x "$PYENV_PYTHON_PATH" ]; then
    echo "⚠️  Python $PYTHON_VERSION no encontrado en pyenv. Reinstalando..."
    pyenv uninstall -f "$PYTHON_VERSION" 2>/dev/null || true
    CPPFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib64" pyenv install "$PYTHON_VERSION"
    pyenv rehash
    pyenv local "$PYTHON_VERSION" 2>/dev/null || true
fi

# Confirmar versión activa usando path absoluto
ACTIVE_PYTHON=$("$PYENV_PYTHON_PATH" --version 2>&1 || echo "ERROR")
echo "   Python activo: $ACTIVE_PYTHON"

# Verificar que es la versión correcta
if [[ "$ACTIVE_PYTHON" != *"$PYTHON_VERSION"* ]]; then
    echo "❌ Error: Python instalado ($ACTIVE_PYTHON) no coincide con requerido ($PYTHON_VERSION)"
    exit 1
fi

# Verificar/instalar Poetry dentro de pyenv (NO usar poetry del sistema)
echo "   Verificando Poetry en pyenv..."
if [ ! -x "$PYENV_POETRY_PATH" ]; then
    echo "   Instalando Poetry en Python $PYTHON_VERSION..."
    "$PYENV_PIP_PATH" install --upgrade pip
    "$PYENV_PIP_PATH" install poetry
    if [ ! -x "$PYENV_POETRY_PATH" ]; then
        echo "❌ Error: No se pudo instalar Poetry en pyenv"
        exit 1
    fi
fi
echo "   ✅ Poetry disponible en: $PYENV_POETRY_PATH"

# Configurar PATH (por si algún subproceso lo necesita)
export PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin:$PATH"

# Verificar GPU si no estamos omitiendo NVIDIA
if [ "$SKIP_NVIDIA" = "false" ] && [[ "$(uname)" == "Linux" ]]; then
    # Asegurar que las variables de entorno de CUDA estén presentes explícitamente antes de arrancar
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda
    
    # Reiniciar nvidia-persistenced si es posible para asegurar estado limpio
    if systemctl list-unit-files | grep -q nvidia-persistenced; then
        echo "🔄 Reiniciando nvidia-persistenced..."
        sudo systemctl restart nvidia-persistenced &> /dev/null || true
    fi

    # Loop de espera para el driver
    MAX_RETRIES=30
    COUNT=0
    while ! nvidia-smi &> /dev/null; do
        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo "⚠️  Timeout esperando a la GPU después de $MAX_RETRIES intentos."
            echo "   El sistema necesita un reinicio adicional."
            ask_reboot "gpu_timeout"
        fi
        echo "   Esperando driver NVIDIA... ($COUNT/$MAX_RETRIES)"
        sleep 2
        COUNT=$((COUNT+1))
    done
    
    if nvidia-smi &> /dev/null; then
        echo "✅ GPU detectada y lista"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        
        # GPU funcionando - resetear contadores de estado
        reset_reboot_count
    fi
fi

# Ejecutar el comando de producción
echo "🚀 Ejecutando aplicación en modo producción..."

# Usar Poetry de pyenv con path absoluto (garantiza Python 3.12.3)
if [ "$ENV_TRUE" = "true" ]; then
    echo "🐍 Modo --env-true: virtualenv en .venv/ dentro del proyecto"
    "$PYENV_POETRY_PATH" config virtualenvs.create true
    "$PYENV_POETRY_PATH" config virtualenvs.in-project true
    echo "📦 Instalando dependencias en virtualenv del proyecto..."
    "$PYENV_POETRY_PATH" install --no-interaction
    echo "✅ Dependencias instaladas en $(pwd)/.venv"
    "$PYENV_POETRY_PATH" run prod
elif [ "$ENV_FALSE" = "true" ]; then
    # Leer CLIP_VENV del .env
    CLIP_VENV=$(grep '^CLIP_VENV=' .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
    CLIP_VENV="${CLIP_VENV:-/opt/subs-gp/CLIP_ENVIRONMENT}"
    echo "🐍 Modo --env-false: virtualenv en $CLIP_VENV"
    # Crear venv si no existe
    if [ ! -f "$CLIP_VENV/bin/python" ]; then
        echo "📦 Creando virtualenv en $CLIP_VENV..."
        "$PYENV_PYTHON_PATH" -m venv "$CLIP_VENV"
        echo "✅ Virtualenv creado en $CLIP_VENV"
    else
        echo "✅ Virtualenv encontrado en $CLIP_VENV"
    fi
    # Ajustar permisos para ec2-user / usuario ejecutor
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$CLIP_VENV" 2>/dev/null || true
    elif id "ec2-user" &>/dev/null; then
        chown -R ec2-user:ec2-user "$CLIP_VENV" 2>/dev/null || true
    fi
    chmod -R 775 "$CLIP_VENV" 2>/dev/null || true

    export VIRTUAL_ENV="$CLIP_VENV"
    export PATH="$CLIP_VENV/bin:$PATH"
    "$PYENV_POETRY_PATH" config virtualenvs.create false
    echo "📦 Instalando dependencias en $CLIP_VENV..."
    "$PYENV_POETRY_PATH" install --no-interaction
    echo "✅ Dependencias instaladas. Ejecutando desde $CLIP_VENV/bin/prod"
    "$CLIP_VENV/bin/prod"
else
    # Modo por defecto: pyenv site-packages compartido
    echo "🐍 Modo por defecto: site-packages de pyenv ($PYENV_PYTHON_PATH)"
    "$PYENV_POETRY_PATH" config virtualenvs.create false
    # poetry install ya se ejecutó en la primera instalación
    "$PYENV_POETRY_PATH" run prod
fi
