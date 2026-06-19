"""
Structural tests for Dockerfile.vllm and related GPU build artefacts.
These tests validate that the GPU container definition is complete and
consistent with the serving code, without actually building the image.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCKERFILE = ROOT / "Dockerfile.vllm"
REQUIREMENTS = ROOT / "app" / "requirements-vllm.txt"
MAKEFILE = ROOT / "Makefile"


# ---------------------------------------------------------------------------
# Dockerfile.vllm existence and structure
# ---------------------------------------------------------------------------

def test_dockerfile_vllm_exists():
    assert DOCKERFILE.exists(), "Dockerfile.vllm is missing"


def test_dockerfile_vllm_uses_cuda_base():
    content = DOCKERFILE.read_text()
    assert "nvidia/cuda" in content, "Dockerfile.vllm must use a CUDA base image"


def test_dockerfile_vllm_entrypoint_uses_main_vllm():
    content = DOCKERFILE.read_text()
    assert "app.main_vllm" in content, "CMD must reference app.main_vllm"


def test_dockerfile_vllm_exposes_8080():
    content = DOCKERFILE.read_text()
    assert "EXPOSE 8080" in content, "Dockerfile.vllm must expose port 8080"


def test_dockerfile_vllm_copies_requirements():
    content = DOCKERFILE.read_text()
    assert "requirements-vllm.txt" in content, "Dockerfile.vllm must copy requirements-vllm.txt"


def test_dockerfile_vllm_copies_app_dir():
    content = DOCKERFILE.read_text()
    assert "COPY app/" in content, "Dockerfile.vllm must copy the app/ directory"


def test_dockerfile_vllm_copies_adapter():
    content = DOCKERFILE.read_text()
    assert "mistral-finetuned" in content, "Dockerfile.vllm must copy the LoRA adapter"


def test_dockerfile_vllm_sets_base_model_env():
    content = DOCKERFILE.read_text()
    assert "BASE_MODEL_ID" in content, "Dockerfile.vllm must set BASE_MODEL_ID env var"


def test_dockerfile_vllm_sets_adapter_path_env():
    content = DOCKERFILE.read_text()
    assert "ADAPTER_PATH" in content, "Dockerfile.vllm must set ADAPTER_PATH env var"


def test_dockerfile_vllm_sets_model_version_env():
    content = DOCKERFILE.read_text()
    assert "MODEL_VERSION" in content, "Dockerfile.vllm must set MODEL_VERSION env var"


def test_dockerfile_vllm_uses_uvicorn():
    content = DOCKERFILE.read_text()
    assert "uvicorn" in content, "CMD must use uvicorn to serve the app"


# ---------------------------------------------------------------------------
# app/requirements-vllm.txt
# ---------------------------------------------------------------------------

def test_requirements_vllm_exists():
    assert REQUIREMENTS.exists(), "app/requirements-vllm.txt is missing"


def test_requirements_vllm_contains_vllm():
    content = REQUIREMENTS.read_text()
    assert "vllm" in content, "requirements-vllm.txt must include the vllm package"


def test_requirements_vllm_contains_fastapi():
    content = REQUIREMENTS.read_text()
    assert "fastapi" in content, "requirements-vllm.txt must include fastapi"


def test_requirements_vllm_contains_uvicorn():
    content = REQUIREMENTS.read_text()
    assert "uvicorn" in content, "requirements-vllm.txt must include uvicorn"


# ---------------------------------------------------------------------------
# Makefile GPU targets
# ---------------------------------------------------------------------------

def test_makefile_has_docker_build_vllm():
    content = MAKEFILE.read_text()
    assert "docker-build-vllm" in content, "Makefile must define a docker-build-vllm target"


def test_makefile_docker_build_vllm_uses_dockerfile_vllm():
    content = MAKEFILE.read_text()
    assert "Dockerfile.vllm" in content, "docker-build-vllm must reference Dockerfile.vllm"


def test_makefile_has_docker_run_vllm():
    content = MAKEFILE.read_text()
    assert "docker-run-vllm" in content, "Makefile must define a docker-run-vllm target"


def test_makefile_docker_run_vllm_uses_mock_mode():
    content = MAKEFILE.read_text()
    assert "MOCK_MODE" in content, "docker-run-vllm must set MOCK_MODE for local testing"
