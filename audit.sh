#!/bin/bash

# Define el nombre del archivo de salida con la fecha y hora actual
OUTPUT_FILE="audit_rhel_$(date +%F_%H-%M).txt"

echo "⏳ Iniciando... Los resultados se mostrarán aquí y se guardarán en: $OUTPUT_FILE"
echo ""

# Todo lo que esté dentro de estas llaves se imprimirá en pantalla y se guardará en el txt
{
echo "================================================"
echo "       HARDENING RHEL / GS         "
echo "================================================"
echo "Descripción: Script para validar configuraciones"
echo "de seguridad, systemd, SELinux y especificaciones."
echo "================================================"
echo ""

echo "================= SYSTEM SPECS ================="
echo "Descripción: Información general del sistema base,"
echo "recursos disponibles y versión de SO."
echo "Operating System:"
grep PRETTY_NAME /etc/os-release | cut -d= -f2 | sed 's/"//g'
echo "------------------------------------------------"
echo "CPU Cores: $(nproc)"
echo "------------------------------------------------"
echo "RAM:"
free -h | awk '/Mem:/ {print "Total:", $2, "| Used:", $3, "| Available:", $7}'
echo "------------------------------------------------"
echo "Disk (/):"
df -h / | awk 'NR==2 {print "Total:", $2, "| Used:", $3, "| Free:", $4}'
echo "================================================"
echo ""

echo "================= SELINUX STATUS ==============="
echo "Descripción: Verifica si el kernel está bloqueando"
echo "la lectura de tu .env o la ejecución de binarios."
if command -v sestatus >/dev/null 2>&1; then
    sestatus | grep -E "SELinux status|Current mode"
else
    echo "SELinux no está instalado o comando no disponible."
fi
echo "================================================"
echo ""

echo "================= SYSTEMD OVERRIDES ============"
echo "Descripción: Busca modificaciones globales de systemd"
echo "que inyectan reglas como ProtectHome=yes impidiendo"
echo "que los servicios lean archivos en directorios de usuario."
systemd-delta --type=extended 2>/dev/null | head -n 15
echo "------------------------------------------------"
echo "(Mostrando los primeros 15 registros globales...)"
echo "================================================"
echo ""

echo "================= SECURITY AGENTS =============="
echo "Descripción: Lista agentes de seguridad (EDR/XDR)"
echo "y políticas de listas blancas (fapolicyd) activos"
echo "que podrían estar matando tu proceso silenciosamente."
systemctl list-units --state=running | grep -iE "fapolicyd|audit|tanium|crowdstrike|falcon|symantec|mcafee|qualys|fireeye" || echo "Ningún agente corporativo listado encontrado."
echo "================================================"
echo ""

echo "================= UMASK & ENV GLOBALES ========="
echo "Descripción: Revisa si hay políticas de permisos"
echo "estrictas (umask) que crean archivos sin permisos"
echo "de lectura para systemd."
echo "Umask en bashrc/profile:"
grep -i umask /etc/bashrc /etc/profile 2>/dev/null | sed 's/^/- /' || echo "- No se encontraron umasks globales"
echo "------------------------------------------------"
echo "Scripts de entorno en /etc/profile.d/:"
ls -la /etc/profile.d/ | awk '{print $9}' | grep -E "\.sh$" | xargs echo
echo "================================================"
echo ""

echo "================= PYTHON & POETRY =============="
echo "Descripción: Info de instalación, paths y configs."
echo "Python3 Path: $(which python3 2>/dev/null || echo 'Not found')"
echo "Python3 Version: $(python3 --version 2>&1 || echo 'Not found')"
echo "Poetry Path: $(which poetry 2>/dev/null || echo 'Not found')"
echo "Poetry Version: $(poetry --version 2>&1 || echo 'Not found')"
echo "------------------------------------------------"
echo "Poetry Env List:"
poetry env list 2>/dev/null || echo "No se pudo listar entornos"
echo "------------------------------------------------"
echo "Poetry Config:"
poetry config --list 2>/dev/null || echo "No se pudo leer config de poetry"
echo "------------------------------------------------"
echo "Environment Variables (PATH & PYTHON):"
env | grep -iE '^PATH=|^PYTHON|^POETRY'
echo "================================================"
echo ""

echo "================= DIRECTORY MOUNTS (noexec) ===="
echo "Descripción: Revisa si particiones como /tmp, /home o /var/tmp"
echo "tienen la bandera 'noexec' previniendo la ejecución de scripts."
mount | grep -iE '/tmp|/home|/var/tmp' | awk '{print $1, $2, $3, $4, $5, $6}'
echo "================================================"
echo ""

echo "================= SELINUX & FILE CONTEXTS ======"
echo "Descripción: Revisa los contextos de seguridad de archivos clave."
echo "Python bin:"
ls -Z $(which python3 2>/dev/null || echo "/usr/bin/python3") 2>/dev/null || echo "No encontrado"
echo "Poetry bin:"
ls -Z $(which poetry 2>/dev/null || echo "/usr/bin/poetry") 2>/dev/null || echo "No encontrado"
echo "~/.cache/pypoetry:"
ls -Zd ~/.cache/pypoetry 2>/dev/null || echo "Carpeta no encontrada o sin acceso"
echo "Current Directory Context:"
ls -Zd .
echo "================================================"
echo ""

echo "================= AUDIT & FAPOLICYD LOGS ======="
echo "Descripción: Muestra eventos recientes de bloqueo (SELinux/Fapolicyd)"
echo "buscando si matan al subproceso de python o poetry."
echo "SELinux Denials (últimos 20):"
sudo ausearch -m AVC,USER_AVC -ts recent 2>/dev/null | tail -n 20 || echo "Sin sudo, command no found, o sin denegaciones."
echo "------------------------------------------------"
echo "Fapolicyd denials (últimos 20):"
sudo journalctl -u fapolicyd --since today --no-pager 2>/dev/null | grep -i deny | tail -n 20 || echo "Sin acceso a fapolicyd logs o sin denegaciones."
echo "================================================"

} | tee "$OUTPUT_FILE"

echo ""
echo "✅ Ejecución finalizada. Puedes revisar el reporte completo ejecutando: cat $OUTPUT_FILE"