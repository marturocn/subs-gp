import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Constants
PYTHON_EXE = "python.exe"
PYTHON_BIN = "python"
SCRIPTS_DIR = "Scripts"
BIN_DIR = "bin"
REQUIRED_PYTHON_VERSION = "3.12"


def get_required_python_version(project_path: Path) -> str:
    """Get the required Python version from .python-version file."""
    python_version_file = project_path / ".python-version"
    if python_version_file.exists():
        version = python_version_file.read_text().strip()
        if not version:
            print("❌ Error: .python-version file is empty")
            raise ValueError("Empty .python-version file")
        print(f"📋 Required Python version from .python-version: {version}")
        return version
    raise FileNotFoundError(f".python-version file not found at {python_version_file}")


def check_pyenv_installed() -> bool:
    """Check if pyenv is installed."""
    result = subprocess.run(
        ['pyenv', '--version'],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def install_pyenv() -> bool:
    """Install pyenv using the official installer."""
    print("📦 Installing pyenv...")
    try:
        # Use the official pyenv installer
        result = subprocess.run(
            ['bash', '-c', 'curl https://pyenv.run | bash'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to install pyenv: {result.stderr}")
            return False
        
        # Add pyenv to PATH for current session
        pyenv_root = Path.home() / ".pyenv"
        os.environ['PYENV_ROOT'] = str(pyenv_root)
        os.environ['PATH'] = f"{pyenv_root}/bin:{pyenv_root}/shims:{os.environ.get('PATH', '')}"
        
        print("✅ pyenv installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing pyenv: {e}")
        return False


def setup_pyenv_environment() -> None:
    """Setup pyenv environment variables for current session."""
    pyenv_root = Path.home() / ".pyenv"
    if pyenv_root.exists():
        os.environ['PYENV_ROOT'] = str(pyenv_root)
        os.environ['PATH'] = f"{pyenv_root}/bin:{pyenv_root}/shims:{os.environ.get('PATH', '')}"


def install_python_version(version: str) -> bool:
    """Install a specific Python version using pyenv."""
    print(f"📦 Installing Python {version} with pyenv...")
    
    # Check if version is already installed
    result = subprocess.run(
        ['pyenv', 'versions', '--bare'],
        capture_output=True,
        text=True
    )
    
    installed_versions = result.stdout.strip().split('\n')
    # Check for exact match or version prefix match (e.g., "3.12" matches "3.12.3")
    version_installed = any(v.startswith(version) for v in installed_versions)
    
    if version_installed:
        print(f"✅ Python {version} is already installed")
        return True
    
    # Install the version
    print(f"⏳ Installing Python {version} (this may take a while)...")
    result = subprocess.run(
        ['pyenv', 'install', version, '-s'],  # -s to skip if already installed
        capture_output=False,  # Show output to user
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to install Python {version}")
        return False
    
    print(f"✅ Python {version} installed successfully")
    return True


def set_pyenv_local(version: str, project_path: Path) -> bool:
    """Set local Python version using pyenv local."""
    print(f"🔧 Setting pyenv local to Python {version}...")
    result = subprocess.run(
        ['pyenv', 'local', version],
        cwd=project_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to set pyenv local: {result.stderr}")
        return False
    
    print(f"✅ pyenv local set to Python {version}")
    return True


def get_pyenv_python_executable(version: str) -> Path | None:
    """Get the Python executable path from pyenv for a specific version."""
    result = subprocess.run(
        ['pyenv', 'prefix', version],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return None
    
    prefix = Path(result.stdout.strip())
    python_exe = prefix / "bin" / "python"
    
    if python_exe.exists():
        return python_exe
    return None


def setup_pyenv(project_path: Path) -> str | None:
    """Setup pyenv and install required Python version. Returns the Python executable path."""
    required_version = get_required_python_version(project_path)
    
    # Setup pyenv environment
    setup_pyenv_environment()
    
    # Check if pyenv is installed
    if not check_pyenv_installed():
        print("⚠️  pyenv is not installed")
        if not install_pyenv():
            print("❌ Could not install pyenv automatically")
            print("   Please install pyenv manually: https://github.com/pyenv/pyenv#installation")
            return None
    
    print("✅ pyenv is available")
    
    # Install required Python version
    if not install_python_version(required_version):
        return None
    
    # Set pyenv local
    if not set_pyenv_local(required_version, project_path):
        return None
    
    # Get the Python executable
    python_exe = get_pyenv_python_executable(required_version)
    if python_exe:
        print(f"✅ Using Python executable: {python_exe}")
        return str(python_exe)
    
    return None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single .env line and return (key, value) or None."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    if '=' not in line:
        return None
    key, value = line.split('=', 1)
    key = key.strip()
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return key, value


def load_env_file(env_file_path: Path) -> None:
    """Load environment variables from a .env file manually."""
    if not env_file_path.exists():
        return

    print(f"📄 Loading environment variables from: {env_file_path}")

    with open(env_file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            parsed = _parse_env_line(raw)
            if not parsed:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value


def get_python_executable(venv_path: Path) -> Path:
    """Get the Python executable from the virtual environment."""
    if os.name == 'nt':
        python_exe = venv_path / SCRIPTS_DIR / PYTHON_EXE
    else:
        python_exe = venv_path / BIN_DIR / PYTHON_BIN
        if not python_exe.exists():
            python_exe = venv_path / BIN_DIR / "python3"
    
    if not python_exe.exists():
        raise FileNotFoundError(f"No Python executable found in {venv_path}")
    
    return python_exe


def create_venv(venv_path: Path, python_exe: str | None = None) -> bool:
    """Create a new virtual environment at the specified path."""
    print(f"📦 Creating virtual environment at: {venv_path}")
    try:
        # Use pyenv Python if available, otherwise fall back to sys.executable
        python_cmd = python_exe if python_exe else sys.executable
        print(f"   Using Python: {python_cmd}")
        
        result = subprocess.run(
            [python_cmd, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Failed to create virtual environment:")
            print(f"   stderr: {result.stderr}")
            return False
        print(f"✅ Virtual environment created successfully at: {venv_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False


def validate_shared_venv(venv_path: Path, python_exe: str | None = None) -> bool:
    """Validate that the shared virtual environment exists and is valid.
    If it doesn't exist, create it automatically."""
    if not venv_path.exists():
        print(f"⚠️  Virtual environment does not exist: {venv_path}")
        if not create_venv(venv_path, python_exe):
            return False
    
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if not pyvenv_cfg.exists():
        print(f"❌ Invalid virtual environment (missing pyvenv.cfg): {venv_path}")
        print("🔄 Recreating virtual environment...")
        # Remove invalid venv and recreate
        import shutil
        shutil.rmtree(venv_path, ignore_errors=True)
        if not create_venv(venv_path, python_exe):
            return False
    
    try:
        venv_python_exe = get_python_executable(venv_path)
        print(f"✅ Valid virtual environment found: {venv_path}")
        print(f"   Python executable: {venv_python_exe}")
        return True
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False


def check_critical_dependencies(venv_path: Path) -> list[str]:
    """Check if critical dependencies are installed. Returns list of missing modules."""
    python_exe = get_python_executable(venv_path)
    critical_modules = ['gunicorn', 'fastapi', 'uvicorn']
    missing = []
    
    for module in critical_modules:
        result = subprocess.run(
            [str(python_exe), "-c", f"import {module}"],
            capture_output=True
        )
        if result.returncode != 0:
            missing.append(module)
    
    return missing


def get_poetry_executable() -> str:
    """Get the Poetry executable path."""
    import shutil
    # Check if poetry is available in PATH (e.g. from pyenv)
    which_poetry = shutil.which("poetry")
    if which_poetry:
        return which_poetry

    # Check pyenv versions directly
    pyenv_root = Path(os.environ.get("PYENV_ROOT", Path.home() / ".pyenv"))
    for pyenv_poetry in pyenv_root.glob("versions/*/bin/poetry"):
        if pyenv_poetry.is_file() and os.access(pyenv_poetry, os.X_OK):
            return str(pyenv_poetry)

    # Common Poetry installation paths
    poetry_paths = [
        Path.home() / ".local" / "bin" / "poetry",
        Path("/usr/local/bin/poetry"),
        Path("/usr/bin/poetry"),
    ]
    
    for poetry_path in poetry_paths:
        if poetry_path.exists():
            return str(poetry_path)
    
    # Fall back to just 'poetry' and hope it's in PATH
    return "poetry"


def check_dependencies_installed(venv_path: Path) -> bool:
    """Check if the main dependencies are already installed in the venv."""
    try:
        python_exe = get_python_executable(venv_path)
    except FileNotFoundError:
        return False
    
    # Check a few key packages that indicate dependencies are installed
    key_packages = ['fastapi', 'uvicorn', 'torch', 'transformers']
    
    for package in key_packages:
        result = subprocess.run(
            [str(python_exe), "-c", f"import {package}"],
            capture_output=True
        )
        if result.returncode != 0:
            return False
    
    return True



def install_dependencies_with_poetry(venv_path: Path, project_path: Path) -> bool:
    """Install dependencies using Poetry into an existing external virtualenv.
    
    According to Poetry docs:
    - Set virtualenvs.create = false to prevent Poetry from creating new venvs
    - Set VIRTUAL_ENV and PATH to "activate" the existing venv
    - Poetry will detect the activated venv and install into it
    
    IMPORTANT: Any old Poetry cached venvs for this project must be removed first,
    otherwise Poetry might get confused about which venv to use.
    """
    poetry_exe = get_poetry_executable()
    if not poetry_exe:
        print("❌ Poetry executable not found")
        return False

    print("📦 Installing dependencies with Poetry...")
    print(f"   Using Poetry: {poetry_exe}")
    print(f"   Target virtualenv: {venv_path}")
    
    venv_python = venv_path / BIN_DIR / "python" if os.name != 'nt' else venv_path / SCRIPTS_DIR / "python.exe"
    venv_bin = venv_path / BIN_DIR if os.name != 'nt' else venv_path / SCRIPTS_DIR
    
    # Verify venv exists and is valid
    if not venv_python.exists():
        print(f"❌ Virtualenv Python not found: {venv_python}")
        return False
    
    # Step 1: Remove any old Poetry cached virtualenvs for this project
    _clean_old_poetry_venvs(poetry_exe, project_path)
    
    # Step 2: Set up environment to "activate" the virtualenv
    install_env = os.environ.copy()
    install_env['VIRTUAL_ENV'] = str(venv_path)
    install_env['PATH'] = f"{venv_bin}:{install_env.get('PATH', '')}"
    install_env['POETRY_VIRTUALENVS_CREATE'] = 'false'
    install_env.pop('PYTHONHOME', None)
    
    # Step 3: Verify Poetry detects our virtualenv correctly
    _verify_poetry_env_detection(poetry_exe, project_path, install_env, venv_path)
    
    # Step 4: Generate lock file if it doesn't exist
    _ensure_lock_file(poetry_exe, project_path, install_env)
    
    # Step 5: Install dependencies using Poetry
    return _run_poetry_install(poetry_exe, project_path, install_env, venv_path)

def _clean_old_poetry_venvs(poetry_exe, project_path):
    """Remove any old Poetry cached virtualenvs for this project."""
    print("🧹 Cleaning old Poetry virtualenv associations...")
    clean_result = subprocess.run(
        [poetry_exe, 'env', 'remove', '--all'],
        cwd=project_path,
        capture_output=True,
        text=True
    )
    if clean_result.returncode == 0 and clean_result.stdout.strip():
        print(f"   {clean_result.stdout.strip()}")

def _verify_poetry_env_detection(poetry_exe, project_path, install_env, venv_path):
    """Verify Poetry detects our virtualenv correctly."""
    print("🔍 Verifying Poetry environment detection...")
    env_info_result = subprocess.run(
        [poetry_exe, 'env', 'info'],
        cwd=project_path,
        capture_output=True,
        text=True,
        env=install_env
    )
    
    if env_info_result.returncode == 0:
        env_info = env_info_result.stdout
        if str(venv_path) in env_info:
            print(f"   ✅ Poetry detected virtualenv: {venv_path}")
        else:
            print(f"   ⚠️ Poetry env info:\n{env_info}")
            print(f"   Expected path: {venv_path}")

def _ensure_lock_file(poetry_exe, project_path, install_env):
    """Generate lock file if it doesn't exist."""
    lock_file = project_path / "poetry.lock"
    if not lock_file.exists():
        print("🔒 Generating poetry.lock...")
        lock_result = subprocess.run(
            [poetry_exe, 'lock'],
            cwd=project_path,
            capture_output=True,
            text=True,
            env=install_env
        )
        if lock_result.returncode != 0:
            print(f"⚠️  Poetry lock failed: {lock_result.stderr}")
    else:
        print("✅ Using existing poetry.lock")

def _run_poetry_install(poetry_exe, project_path, install_env, venv_path):
    """Run poetry install command."""
    print(f"📥 Installing dependencies into {venv_path}...")
    
    result = subprocess.run(
        [poetry_exe, 'install', '--only=main', '--no-interaction', '--no-root'],
        cwd=project_path,
        capture_output=True,
        text=True,
        env=install_env
    )
    
    if result.returncode != 0:
        print("❌ Poetry install failed:")
        print(f"   stdout: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
        print(f"   stderr: {result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr}")
        return False
    
    if result.stdout.strip():
        # Show just last few lines of output
        lines = result.stdout.strip().split('\n')
        if len(lines) > 5:
            print(f"   ... ({len(lines)} packages processed)")
        print(f"   {lines[-1] if lines else ''}")
    print("✅ Dependencies installed successfully with Poetry")
    return True

def setup_shared_venv(pyenv_python: str | None = None) -> Dict[str, str]:
    """Set up and validate the shared virtual environment specified by CLIP_VENV."""
    clip_venv = os.getenv("CLIP_VENV")
    
    if not clip_venv:
        raise ValueError("❌ Environment variable CLIP_VENV must be defined")
    
    venv_path = Path(clip_venv)
    project_path = Path(__file__).parent.parent
    
    # Validate the shared virtual environment exists (creates it if not)
    if not validate_shared_venv(venv_path, pyenv_python):
        raise RuntimeError(
            f"❌ Could not create or validate virtual environment at: {venv_path}"
        )
    
    # Always install dependencies with Poetry
    if not install_dependencies_with_poetry(venv_path, project_path):
        raise RuntimeError("❌ Failed to install dependencies with Poetry")
    
    # Prepare environment for running the application
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(venv_path)
    env['POETRY_VIRTUALENVS_CREATE'] = 'false'
    env['POETRY_VIRTUALENVS_IN_PROJECT'] = 'false'
    
    if os.name == 'nt':
        env['PATH'] = f"{venv_path / SCRIPTS_DIR}{os.pathsep}{env.get('PATH', '')}"
    else:
        env['PATH'] = f"{venv_path / BIN_DIR}{os.pathsep}{env.get('PATH', '')}"
    
    print(f"🚀 Using shared virtual environment: {venv_path}")
    
    return env


def main() -> None:
    """Main entry point for production server."""
    print("=" * 60)
    print("🚀 Starting Production Server")
    print("=" * 60)
    
    # Get the project root directory
    project_path = Path(__file__).parent.parent.resolve()
    
    # Load environment variables from the local .env file BEFORE anything else
    local_env_file = project_path / ".env"
    load_env_file(local_env_file)
    
    # Setup pyenv and get the correct Python executable
    print("\n🐍 Setting up Python environment with pyenv...")
    pyenv_python = setup_pyenv(project_path)
    if not pyenv_python:
        print("⚠️  Could not setup pyenv, will try with system Python")
    
    # Set up and validate the shared virtual environment
    env = setup_shared_venv(pyenv_python)
    
    # Load environment variables from the virtual environment directory (if exists)
    clip_venv = os.getenv("CLIP_VENV")
    if clip_venv:
        env_file = Path(clip_venv) / ".env"
        load_env_file(env_file)

    # Calculate workers based on CPU count
    cpu_count = os.cpu_count() or 2
    num_workers = int(os.getenv('API_WORKERS', max(1, cpu_count // 2)))
    api_port = os.getenv('API_PORT', '8000')
    
    print("🔧 Configuration:")
    print(f"   - Workers: {num_workers}")
    print(f"   - Port: {api_port}")
    print(f"   - Virtual Environment: {clip_venv}")
    print(f"   - Project Path: {project_path}")
    print("=" * 60)

    # Get the gunicorn executable from the virtual environment
    if not clip_venv:
        raise RuntimeError("❌ CLIP_VENV environment variable is not set")
    
    # Set PYTHONPATH to prioritize the current project directory
    # This is critical when multiple projects share the same virtual environment
    project_src = str(project_path / "src")
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{project_src}{os.pathsep}{str(project_path)}{os.pathsep}{current_pythonpath}"
    else:
        env['PYTHONPATH'] = f"{project_src}{os.pathsep}{str(project_path)}"
    
    print(f"   - PYTHONPATH: {env['PYTHONPATH']}")

    # Run toOnnx before starting the app
    print("🔄 Running toOnnx conversion...")
    poetry_exe = get_poetry_executable()
    onnx_result = subprocess.run(
        [poetry_exe, 'run', 'toOnnx'],
        cwd=project_path,
        env=env
    )
    if onnx_result.returncode != 0:
        print("⚠️  toOnnx conversion failed, continuing anyway...")
    else:
        print("✅ toOnnx conversion completed")
    
    # Run uvicorn directly
    cmd = [
        "uvicorn",
        "src.main:app",
        "--host", "0.0.0.0",
        "--port", str(os.getenv('API_PORT', 8000))
    ]
    
    print(f"▶️  Executing: {' '.join(cmd)}")
    
    # Execute with the configured environment from the project directory
    sys.exit(subprocess.call(cmd, env=env, cwd=project_path))

if __name__ == "__main__":
    main()
