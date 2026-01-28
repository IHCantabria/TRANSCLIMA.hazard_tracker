""" Template """
import tomli
# Versión del paquete
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

version = config["project"]["version"]