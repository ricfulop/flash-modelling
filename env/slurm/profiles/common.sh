#!/usr/bin/env bash
# Resolve flash-modelling repo root on either DGX node.
set -euo pipefail

if [[ -n "${FLASH_MODELLING_ROOT:-}" && -d "${FLASH_MODELLING_ROOT}" ]]; then
  export FLASH_MODELLING_ROOT
  return 0 2>/dev/null || exit 0
fi

case "$(hostname -s)" in
  spark-5da5)
    export FLASH_MODELLING_ROOT="/home/ricfulop/Desktop/Cursor/flash-modelling"
    ;;
  spark-0808)
    export FLASH_MODELLING_ROOT="/home/nvidia/Cursor/flash-modelling"
    ;;
  *)
    if [[ -d "/home/ricfulop/Desktop/Cursor/flash-modelling" ]]; then
      export FLASH_MODELLING_ROOT="/home/ricfulop/Desktop/Cursor/flash-modelling"
    elif [[ -d "/home/nvidia/Cursor/flash-modelling" ]]; then
      export FLASH_MODELLING_ROOT="/home/nvidia/Cursor/flash-modelling"
    else
      echo "ERROR: could not locate flash-modelling checkout on $(hostname)" >&2
      return 1 2>/dev/null || exit 1
    fi
    ;;
esac

export CURSOR_ROOT="$(dirname "${FLASH_MODELLING_ROOT}")"
