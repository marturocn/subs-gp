#!/bin/bash

set -e

# =============================================================================
# PROCESAMIENTO DE ARGUMENTOS
# =============================================================================

RHEL_MODE=false
ENV_TRUE=false
ENV_FALSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "  --hat          Usar comandos RHEL/CentOS (dnf) en lugar de Ubuntu (apt)"
            echo "  --env-true     Usar virtualenv por proyecto (virtualenvs.in-project)"
            echo "  --env-false    Usar virtualenv en CLIP_VENV (ruta del .env o /opt/subs-gp/CLIP_ENVIRONMENT)"
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

PYTHON_VERSION=$(cat .python-version)
PYENV_ROOT="$HOME/.pyenv"

echo "🔍 Verificando entorno..."

# =============================================================================
# FUNCIONES DE VERIFICACIÓN E INSTALACIÓN
# =============================================================================

# Verificar si las dependencias de compilación están instaladas
check_build_deps_installed() {
    if [ "$RHEL_MODE" = "true" ]; then
        # Verificar paquetes esenciales en RHEL
        rpm -q gcc make openssl-devel zlib-devel libffi-devel git curl bzip2-devel &> /dev/null
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
    
    # Instalar dependencias de compilación antes de pyenv
    install_pyenv_build_deps
    
    curl -fsSL https://pyenv.run | bash
    
    export PYENV_ROOT
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init -)"
    
    echo "✅ pyenv instalado correctamente"
fi

# ============================================================================
# INSTALAR PYTHON Y POETRY
# ============================================================================

# 2. Verificar Python
echo "🐍 Verificando Python setup..."

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

# Configurar Poetry para usar este Python específico
echo "   Configurando Poetry para usar Python $PYTHON_VERSION..."
export PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin:$PATH"

# Verificar GPU si no estamos omitiendo NVIDIA
if [ "${SKIP_NVIDIA:-false}" = "false" ] && [[ "$(uname)" == "Linux" ]]; then
    # Asegurar que las variables de entorno de CUDA estén presentes explícitamente antes de arrancar
    export PATH=/usr/local/cuda/bin:$PATH
fi

# 3. Verificar Poetry en pyenv
echo "   Verificando Poetry en pyenv..."
if [ ! -x "$PYENV_POETRY_PATH" ]; then
    echo "⚠️  Poetry no encontrado en pyenv. Instalando con pip de pyenv..."
    
    # Instalar Poetry usando el pip del pyenv Python específico
    "$PYENV_PIP_PATH" install poetry
    
    # Verificar que Poetry se instaló correctamente
    if [ -x "$PYENV_POETRY_PATH" ]; then
        echo "✅ Poetry instalado correctamente en pyenv"
    else
        echo "❌ Error: Poetry no se encuentra en $PYENV_POETRY_PATH tras instalación."
        exit 1
    fi
else
    echo "✅ Poetry disponible en $PYENV_POETRY_PATH"
fi

# Verificar configuración del archivo .env
echo "🔧 Verificando archivo .env..."
if [ -f .env ]; then
    echo "✅ Archivo .env encontrado. Asumiendo configuración correcta."
else
    echo "⚠️  Archivo .env no encontrado"
    if [ -f .env.prod ]; then
        echo "📋 Creando .env a partir de .env.prod..."
        cp .env.prod .env
        echo "✅ Archivo .env creado correctamente"
    else
        echo "❌ No se encontró .env ni .env.prod"
        exit 1
    fi
fi

# Ejecutar el comando de producción
echo "🚀 Ejecutando aplicación en modo producción..."

# Usar poetry con el Python correcto (path absoluto, no shims)
# Usar el binario de Poetry instalado en pyenv para asegurar que use Python 3.12.3
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
    echo "📦 Instalando dependencias..."
    "$PYENV_POETRY_PATH" install --no-interaction
    "$PYENV_POETRY_PATH" run prod
fi

