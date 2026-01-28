import argparse
import logging
import subprocess
import sys
from pathlib import Path

try:
    import tomli
    import tomli_w
except ImportError:
    print("❌ Error: tomli/tomli_w libraries not installed")
    print("Install with: pip install tomli tomli-w")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, capture_output=True, check=True):
    """
    Run a shell command using subprocess.

    Args:
        cmd: Command as string or list
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit

    Returns:
        CompletedProcess instance

    Raises:
        subprocess.CalledProcessError if command fails and check=True
    """
    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                check=check,
            )
        else:
            result = subprocess.run(
                cmd, capture_output=capture_output, text=True, check=check
            )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        raise


def parse_version(version_str):
    """Parse version string into components."""
    # Remove 'v' prefix if present
    clean_version = version_str.lstrip("v")
    parts = clean_version.split(".")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid version format: {version_str}. Expected format: v1.2.3"
        )

    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"Invalid version format: {version_str}. Version parts must be integers."
        )


def format_version(major, minor, patch):
    """Format version components into string."""
    return f"v{major}.{minor}.{patch}"


def update_version(current_version, bump_type):
    """
    Update version based on bump type.

    Args:
        current_version: Current version string (e.g., 'v1.2.3')
        bump_type: One of 'major', 'minor', 'patch'

    Returns:
        New version string
    """
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    return format_version(major, minor, patch)


def check_git_status():
    """Check if repository is clean."""
    logger.info("Verificando estado del repositorio...")
    try:
        result = run_command(
            ["git", "diff-index", "--quiet", "HEAD", "--"], check=False
        )
        if result.returncode != 0:
            logger.error("Hay cambios sin commit en el repositorio")
            logger.error(
                "Por favor, haz commit de tus cambios antes de desplegar"
            )
            return False
    except Exception as e:
        logger.error(f"Error verificando estado de git: {e}")
        return False

    logger.info("✓ Repositorio limpio")
    return True


def check_current_branch():
    """Check current git branch."""
    logger.info("Verificando rama actual...")
    try:
        result = run_command(["git", "branch", "--show-current"])
        branch = result.stdout.strip()
        logger.info(f"Rama actual: {branch}")

        if branch not in ["main", "master"]:
            logger.warning(
                f"⚠️  No estás en la rama main/master (estás en '{branch}')"
            )
            response = input("¿Deseas continuar de todas formas? (s/N): ")
            if response.lower() not in ["s", "si", "sí", "yes", "y"]:
                logger.info("Despliegue cancelado")
                return False

        return True
    except Exception as e:
        logger.error(f"Error verificando rama: {e}")
        return False


def run_tests():
    """Run test suite."""
    logger.info("Ejecutando tests...")
    try:
        run_command(["pytest"], capture_output=False)
        logger.info("✓ Tests pasaron exitosamente")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Tests fallaron")
        logger.error("Por favor, arregla los tests antes de desplegar")
        return False


def rollback_changes(original_content, filepath):
    """Rollback changes to pyproject.toml."""
    logger.warning("Realizando rollback de cambios...")
    try:
        with open(filepath, "wb") as f:
            tomli_w.dump(original_content, f)
        logger.info("✓ Rollback completado")
    except Exception as e:
        logger.error(f"Error durante rollback: {e}")
        logger.error("Por favor, revisa manualmente el archivo pyproject.toml")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Script de despliegue para actualizar versión y crear tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python deploy.py --patch    # Incrementa versión patch (v1.2.3 -> v1.2.4)
  python deploy.py --minor    # Incrementa versión minor (v1.2.3 -> v1.3.0)
  python deploy.py --major    # Incrementa versión major (v1.2.3 -> v2.0.0)
  python deploy.py --dry-run --patch  # Simula sin hacer cambios
        """,
    )

    # Create mutually exclusive group for version bump
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument(
        "--major", action="store_true", help="Incrementar versión major"
    )
    version_group.add_argument(
        "--minor", action="store_true", help="Incrementar versión minor"
    )
    version_group.add_argument(
        "--patch", action="store_true", help="Incrementar versión patch"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin hacer cambios reales",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Saltar ejecución de tests"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Confirmar automáticamente"
    )

    return parser.parse_args()


def determine_bump_type(args):
    """Determine version bump type from arguments."""
    if args.major:
        return "major"
    if args.minor:
        return "minor"
    return "patch"


def perform_pre_deploy_checks(skip_tests):
    """Perform pre-deployment checks.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    if not check_git_status():
        return False

    if not check_current_branch():
        return False

    if skip_tests:
        logger.warning("⚠️  Saltando tests (--skip-tests)")
        return True

    return run_tests()


def get_version_info(bump_type):
    """Read current version and calculate new version.

    Returns:
        tuple: (current_version, new_version, original_config, pyproject_path)
    """
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        logger.error("No se encontró pyproject.toml")
        return None

    try:
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)

        original_config = config.copy()
        current_version = config["project"]["version"]
        new_version = update_version(current_version, bump_type)

        logger.info(f"Versión actual: {current_version}")
        logger.info(f"Nueva versión: {new_version}")

        return (
            current_version,
            new_version,
            original_config,
            pyproject_path,
            config,
        )

    except Exception as e:
        logger.error(f"Error leyendo pyproject.toml: {e}")
        return None


def confirm_deployment(new_version, auto_confirm):
    """Ask user to confirm deployment.

    Returns:
        bool: True if confirmed, False otherwise
    """
    if auto_confirm:
        return True

    logger.info(f"\n📦 Se va a desplegar la versión {new_version}")
    logger.info("Esto incluye:")
    logger.info("  1. Actualizar pyproject.toml")
    logger.info("  2. Crear commit con los cambios")
    logger.info(f"  3. Crear tag {new_version}")
    logger.info("  4. Hacer push a origin")

    response = input("\n¿Deseas continuar? (s/N): ")
    return response.lower() in ["s", "si", "sí", "yes", "y"]


def show_dry_run_summary(new_version):
    """Show what would be done in a real deployment."""
    logger.info("\n✓ Dry-run completado. En modo normal se ejecutaría:")
    logger.info(f"  - Actualizar versión a {new_version}")
    logger.info("  - git add pyproject.toml")
    logger.info("  - git commit -m 'Bump version to {new_version}'")
    logger.info("  - git tag -a {new_version} -m 'Version {new_version}'")
    logger.info("  - git push origin {new_version}")
    logger.info("  - git push")


def get_commit_message(new_version, auto_confirm):
    """Get commit message, optionally with user customization.

    Returns:
        str: Commit message
    """
    commit_message = f"Bump version to {new_version}"

    if auto_confirm:
        return commit_message

    response = input(
        "\n¿Deseas añadir un mensaje personalizado al commit? (s/N): "
    )
    if response.lower() not in ["s", "si", "sí", "yes", "y"]:
        return commit_message

    custom_msg = input(
        "Escribe tu mensaje (se añadirá la versión al final): "
    ).strip()

    if custom_msg:
        return f"{custom_msg} - {new_version}"

    return commit_message


def perform_deployment(
    config, new_version, commit_message, pyproject_path, original_config
):
    """Perform the actual deployment operations.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Update version in config
        config["project"]["version"] = new_version

        # Write new version
        logger.info("Actualizando pyproject.toml...")
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(config, f)

        # Git operations
        logger.info("Agregando cambios a git...")
        run_command(["git", "add", "pyproject.toml"])

        logger.info(f"Creando commit: {commit_message}")
        run_command(["git", "commit", "-m", commit_message])

        logger.info(f"Creando tag {new_version}...")
        run_command(
            ["git", "tag", "-a", new_version, "-m", f"Version {new_version}"]
        )

        logger.info("Haciendo push del tag...")
        run_command(["git", "push", "origin", new_version])

        logger.info("Haciendo push de los cambios...")
        run_command(["git", "push"])

        logger.info(f"\n✅ Despliegue completado exitosamente: {new_version}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Error durante el despliegue: {e}")
        rollback_changes(original_config, pyproject_path)
        logger.error(
            "\nPuede que necesites revertir cambios de git manualmente:"
        )
        logger.error("  git reset --soft HEAD~1  # Deshacer último commit")
        logger.error(f"  git tag -d {new_version}  # Eliminar tag local")
        return False

    except Exception as e:
        logger.error(f"\n❌ Error inesperado: {e}")
        rollback_changes(original_config, pyproject_path)
        return False


def main():
    """Main deployment function."""
    args = parse_arguments()
    bump_type = determine_bump_type(args)

    if args.dry_run:
        logger.info("🔍 MODO DRY-RUN: No se realizarán cambios reales")

    # Perform pre-deployment checks
    if not perform_pre_deploy_checks(args.skip_tests):
        sys.exit(1)

    # Get version information
    version_info = get_version_info(bump_type)
    if not version_info:
        sys.exit(1)

    _, new_version, original_config, pyproject_path, config = version_info

    # Handle dry-run mode
    if args.dry_run:
        show_dry_run_summary(new_version)
        sys.exit(0)

    # Confirm deployment with user
    if not confirm_deployment(new_version, args.yes):
        logger.info("Despliegue cancelado")
        sys.exit(0)

    # Get commit message
    commit_message = get_commit_message(new_version, args.yes)

    # Perform the deployment
    if not perform_deployment(
        config, new_version, commit_message, pyproject_path, original_config
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
