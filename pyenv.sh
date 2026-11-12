#!/bin/bash
# =============================================================================
# SCRIPT DE REPARACIÓN: corrige _bz2 cuando pyenv compiló Python sin bzip2
# Uso: sudo ./pyenv.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Auto-detect service user
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

echo -e "${BLUE}ℹ️  SERVICE_USER=$SERVICE_USER  USER_HOME=$USER_HOME${NC}"

# Verificar que los headers de bzip2 existen
echo -e "${BLUE}ℹ️  Verificando headers bzip2...${NC}"
if ! ls /usr/include/bzlib.h /usr/lib64/libbz2* > /dev/null 2>&1; then
    echo -e "${RED}❌ Headers bzip2 no encontrados. Instala bzip2-devel primero:${NC}"
    echo "   dnf install -y bzip2-devel"
    exit 1
fi
echo -e "${GREEN}✅ Headers bzip2 OK${NC}"

# Detectar PYENV_ROOT
PYENV_ROOT="${USER_HOME}/.pyenv"
if [ ! -d "$PYENV_ROOT" ]; then
    PYENV_ROOT="$(find "$USER_HOME" -maxdepth 5 -name 'pyenv' -type d 2>/dev/null | grep -v shims | head -1 || true)"
fi

if [ -z "$PYENV_ROOT" ] || [ ! -d "$PYENV_ROOT" ]; then
    echo -e "${RED}❌ No se encontró pyenv en $USER_HOME${NC}"
    exit 1
fi

# Detectar versión de Python
PYVER="$(cat /opt/subs-gp/clip-vector-api/.python-version 2>/dev/null || cat ./clip-vector-api/.python-version 2>/dev/null || true)"
if [ -z "$PYVER" ]; then
    echo -e "${RED}❌ No se pudo detectar la versión de Python desde /opt/subs-gp/clip-vector-api/.python-version${NC}"
    exit 1
fi

echo -e "${BLUE}ℹ️  PYENV_ROOT=$PYENV_ROOT  PYVER=$PYVER${NC}"

# Función para correr comandos como SERVICE_USER
run_as_user() {
    if [ "$SERVICE_USER" = "root" ] || [ "$(id -un)" = "$SERVICE_USER" ]; then
        bash -c "$1"
    else
        runuser -l "$SERVICE_USER" -c "$1"
    fi
}

# Desinstalar versión actual y reinstalar con CPPFLAGS/LDFLAGS correctos
echo -e "${BLUE}ℹ️  Reinstalando Python $PYVER con soporte bzip2...${NC}"
run_as_user "
    export PYENV_ROOT='$PYENV_ROOT'
    export PATH='\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH'
    pyenv uninstall -f '$PYVER' 2>/dev/null || true
    CPPFLAGS='-I/usr/include' LDFLAGS='-L/usr/lib64' pyenv install '$PYVER'
    pyenv rehash
"

# Verificar que _bz2 compila correctamente
echo -e "${BLUE}ℹ️  Verificando módulo _bz2...${NC}"
run_as_user "
    export PYENV_ROOT='$PYENV_ROOT'
    export PATH='\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH'
    pyenv exec python -c 'import _bz2; print(\"✅ _bz2 OK\")'
"

# Recrear virtualenv y reinstalar dependencias
SVC_PATH="/opt/subs-gp/clip-vector-api"
if [ -d "$SVC_PATH" ]; then
    echo -e "${BLUE}ℹ️  Recreando virtualenv en $SVC_PATH...${NC}"
    rm -rf "${SVC_PATH}/.venv"
    run_as_user "
        export PYENV_ROOT='$PYENV_ROOT'
        export PATH='\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH'
        cd '$SVC_PATH' && poetry install
    "
    echo -e "${GREEN}✅ Dependencias reinstaladas${NC}"
fi

# Reiniciar servicio
echo -e "${BLUE}ℹ️  Reiniciando ire-ei.service...${NC}"
systemctl restart ire-ei.service
echo -e "${GREEN}✅ Servicio reiniciado${NC}"
echo ""
journalctl -u ire-ei.service -n 20 --no-pager
