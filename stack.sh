#!/usr/bin/env bash
#
# Launcher for the Smart Semantic Searcher stack. stack.ps1 is the other
# supported entry point and is kept at parity with this one; scripts/README.md is
# the operator reference for both.
#
# This script parses arguments and runs processes. It does not decide what
# Compose is asked to do -- scripts/stack_core.py does, once, for both
# launchers. The one deliberate exception is the operational recovery commands
# (down, logs, status, help), which never call the core: stopping a stack has to
# keep working with no Python interpreter on PATH and with .env absent or
# malformed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
COMPOSE_FILE="$ROOT_DIR/compose.yml"
EMBEDDER_PRIVATE_COMPOSE_FILE="$ROOT_DIR/compose.embedder-private.yml"
GPU_DEVICE_COMPOSE_FILE="$ROOT_DIR/compose.gpu-device.yml"
ENV_FILE="$ROOT_DIR/.env"
ENV_EXAMPLE="$ROOT_DIR/.env.example"

MINIMUM_COMPOSE_MAJOR=2
MINIMUM_COMPOSE_MINOR=24
MINIMUM_COMPOSE_PATCH=4

# Date-based, and identical in stack.ps1. It answers "are these two launchers
# the same generation", which is the question worth asking when a Windows and a
# Unix operator disagree about what the stack does; a parity test asserts the
# two strings match.
LAUNCHER_VERSION="2026.09.08.1"

usage() {
    cat <<EOF
Smart Semantic Searcher stack launcher $LAUNCHER_VERSION

Examples:
  bash stack.sh up cpu                        build and start the CPU stack
  bash stack.sh up gpu advanced --wait         start the GPU stack on the advanced models, wait for health
  bash stack.sh status cpu                    show what is running
  bash stack.sh logs cpu --service middleware --tail 200
  bash stack.sh urls                          print the endpoints and their authentication state
  bash stack.sh down                          stop the stack

Commands:
  up [cpu|gpu|auto] [basic|advanced]   verify the configuration, build the images, start the stack
  verify                          check the configuration against the cross-service invariants only
  build [cpu|gpu|auto]            build the images and start nothing
  pull [cpu|gpu|auto]             fetch the images this stack does not build itself
  restart [cpu|gpu]               restart the running services without rebuilding
  config [cpu|gpu|auto]           print the resolved Compose configuration, credentials redacted
  urls                            print the published endpoints and each service's auth state
  status [cpu|gpu]                show container status
  logs [cpu|gpu]                  show container logs
  down                            stop and remove the containers
  version                         report the launcher, Docker and Compose versions
  help                            print this text

Options, listed under the command that accepts them:
  up      --no-build  --wait  --wait-timeout <seconds>  --skip-verify
  up      --force-autotune  request one forced preparation before normal startup
  up/config  --autotune <off|quick|full>  --frontend / --no-frontend
  config  --show-secrets
  up/config/build/pull  --gpu-device <device>  select one GPU by index or GPU- UUID/prefix
  logs    --tail <lines>  --follow  --since <time>  --service <name>
  down    --volumes  --remove-orphans  --yes
  every   --dry-run  --env-file <path>  --project-name <name>

Launcher switches, each overriding its configured value for this run:
  --embedder-only / --full                  EMBEDDER_ONLY
  --no-mcp / --mcp                          MCP_SERVER_ENABLED
  --private-embedder / --public-embedder    EXPOSE_SMART_EMBEDDER_PORT
  Passing both halves of a pair is a usage error, not last one wins.

Precedence, highest first: the command line, the shell environment, the
environment file, the built-in default.

Exit codes:
  0    success
  1    unmet prerequisite: Docker missing, daemon down, Compose too old, no environment file
  2    usage error, or a failed configuration preflight
       PowerShell native binding errors (missing/duplicate values) exit 1 before the script
  *    anything else is Docker's own exit code, passed through unchanged

scripts/README.md is the full operator reference for both launchers.
EOF
}

fail_usage() {
    echo "[ERROR] $1" >&2
    usage >&2
    exit 2
}

lowercase() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Keys whose value is treated as a credential by `config`. Compose resolves the
# whole environment, so without this filter `config` printed every bearer token
# in the stack to stdout, unwarned.
SECRET_KEY_PATTERN='[A-Za-z_][A-Za-z0-9_]*_(TOKEN|TOKENS|SECRET|PASSWORD|KEY)'

# Redact credential values from `docker compose config` output.
#
# Two YAML shapes have to survive: the `KEY: value` mapping entries Compose
# emits today, and the `- KEY=value` list entries it accepts on input and may
# emit for an unresolved service. `<redacted>` is a plain YAML scalar -- `<` is
# not an indicator character -- so the output stays parseable.
#
# The key pattern deliberately over-matches. Empty and null values stay visible
# because they prove a variable is unset. Numeric values stay visible only for
# these known batch-size settings; a numeric bearer token is still secret.
#
#   empty ("" or '')  keeping it shows the variable is unset, which is not a
#                     secret: the console already leaks whether auth is required
#                     through /assets/config.js, and blanking it would make
#                     `config` useless for "did my token reach the container".
#   null              same reasoning.
#   an integer        only MILCO_MAX_BATCH_TOKENS,
#                     QWEN_EMBED_MAX_BATCH_TOKENS and
#                     MILCO_PROJECTION_CHUNK_TOKENS are batch sizes.
redact_compose_secrets() {
    sed -E \
        -e "/^[[:space:]]*(-[[:space:]]*)?${SECRET_KEY_PATTERN}[:=][[:space:]]*(\"\"|''|null)?[[:space:]]*\$/b" \
        -e "/^[[:space:]]*(-[[:space:]]*)?(MILCO_MAX_BATCH_TOKENS|QWEN_EMBED_MAX_BATCH_TOKENS|MILCO_PROJECTION_CHUNK_TOKENS)[:=][[:space:]]*\"?[0-9]+\"?[[:space:]]*\$/b" \
        -e "s/^([[:space:]]*)(${SECRET_KEY_PATTERN}):[[:space:]].*\$/\1\2: <redacted>/" \
        -e "s/^([[:space:]]*-[[:space:]]*)(${SECRET_KEY_PATTERN})=.*\$/\1\2=<redacted>/"
}

find_python() {
    local candidate
    for candidate in python3 python py; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

# The env file holds bearer tokens, so a group- or world-readable mode is worth
# a warning -- but only a warning, because tightening it is the operator's call
# and refusing to start over a file mode would be worse than the exposure.
# Skipped on Windows, where Git for Windows synthesises a mode that says nothing
# about the real ACL and would warn on every single run.
warn_insecure_env_permissions() {
    case "${OSTYPE:-}" in
        msys*|cygwin*|win32*) return 0 ;;
    esac
    command -v stat >/dev/null 2>&1 || return 0
    local mode=""
    mode="$(stat -c '%a' "$runtime_env_file" 2>/dev/null)" \
        || mode="$(stat -f '%Lp' "$runtime_env_file" 2>/dev/null)" \
        || return 0
    [[ "$mode" =~ ^[0-7]+$ ]] || return 0
    if [[ "${mode: -2}" != "00" ]]; then
        echo "[WARN] $runtime_env_file is mode $mode and holds bearer tokens; run 'chmod 600' on it." >&2
    fi
}

# Compare a Compose version string against the required minimum numerically.
# Lexical comparison gets 2.100.1 wrong -- "100" sorts before "24" as text --
# and that is exactly the range Compose is in now.
compose_version_supported() {
    local version="${1#v}"
    [[ "$version" =~ ^([0-9]+)(\.([0-9]+))?(\.([0-9]+))?([-+].*)?$ ]] || return 1
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[3]:-0}"
    local patch="${BASH_REMATCH[5]:-0}"
    # Base 10 forced: a component like 09 is an invalid octal literal otherwise.
    major=$((10#$major))
    minor=$((10#$minor))
    patch=$((10#$patch))
    ((major > MINIMUM_COMPOSE_MAJOR)) && return 0
    ((major < MINIMUM_COMPOSE_MAJOR)) && return 1
    ((minor > MINIMUM_COMPOSE_MINOR)) && return 0
    ((minor < MINIMUM_COMPOSE_MINOR)) && return 1
    ((patch >= MINIMUM_COMPOSE_PATCH))
}

# --- argument surface ------------------------------------------------------
#
# One table says which options each command accepts, walked by one loop. The
# alternative -- a per-command validation block -- was five near-identical
# blocks that drifted apart, and its PowerShell twin could not use parameter
# sets for the same job without turning a usage error into a binder failure.
#
# --dry-run, --env-file and --project-name are global, so every command lists
# them; the switch overrides are listed by the commands whose plan they change,
# which includes verify because they change what its token preflight checks.
GLOBAL_OPTIONS="dry-run env-file project-name"
SWITCH_OPTIONS="embedder-only full no-mcp mcp private-embedder public-embedder"
allowed_options_for() {
    case "$1" in
        up)
            printf '%s' "hardware embedder-profile gpu-device no-build wait wait-timeout skip-verify autotune force-autotune frontend no-frontend $SWITCH_OPTIONS $GLOBAL_OPTIONS"
            ;;
        verify) printf '%s' "$SWITCH_OPTIONS $GLOBAL_OPTIONS" ;;
        config)
            printf '%s' "hardware embedder-profile gpu-device show-secrets autotune frontend no-frontend $SWITCH_OPTIONS $GLOBAL_OPTIONS"
            ;;
        build)   printf '%s' "hardware embedder-profile gpu-device $SWITCH_OPTIONS $GLOBAL_OPTIONS" ;;
        pull)    printf '%s' "hardware embedder-profile gpu-device $SWITCH_OPTIONS $GLOBAL_OPTIONS" ;;
        restart) printf '%s' "hardware embedder-profile $SWITCH_OPTIONS $GLOBAL_OPTIONS" ;;
        urls)    printf '%s' "hardware embedder-profile $SWITCH_OPTIONS $GLOBAL_OPTIONS" ;;
        logs)   printf '%s' "hardware tail follow service since $GLOBAL_OPTIONS" ;;
        status) printf '%s' "hardware $GLOBAL_OPTIONS" ;;
        down)   printf '%s' "volumes remove-orphans yes $GLOBAL_OPTIONS" ;;
        version) printf '%s' "" ;;
        help)   printf '%s' "" ;;
        *)      return 1 ;;
    esac
}

# Print one step the launcher would run. One argument per line, so an argument
# containing a space is unambiguous, and so the two launchers can be compared
# byte for byte.
dry_run_index=0
print_dry_run_step() {
    local description="$1"
    shift
    dry_run_index=$((dry_run_index + 1))
    printf 'step %d: %s\n' "$dry_run_index" "$description"
    local argument
    for argument in "$@"; do
        printf '  %s\n' "$argument"
    done
}

option_display_name() {
    case "$1" in
        hardware)         printf '%s' "a hardware profile" ;;
        embedder-profile) printf '%s' "a Smart Embedder profile" ;;
        no-build)         printf '%s' "--no-build" ;;
        wait)             printf '%s' "--wait" ;;
        skip-verify)      printf '%s' "--skip-verify" ;;
        show-secrets)     printf '%s' "--show-secrets" ;;
        tail)             printf '%s' "--tail" ;;
        wait-timeout)     printf '%s' "--wait-timeout" ;;
        *)                printf -- '--%s' "$1" ;;
    esac
}

command_name="${1:-help}"
if (($#)); then
    shift
fi
command_name="$(lowercase "$command_name")"
if [[ "$command_name" == "-h" || "$command_name" == "--help" ]]; then
    command_name="help"
fi

positionals=()
no_build=false
wait_for_health=false
skip_verify=false
show_secrets=false
dry_run=false
embedder_only_flag=false
full_flag=false
no_mcp_flag=false
mcp_flag=false
private_embedder_flag=false
public_embedder_flag=false
follow_logs=false
remove_volumes=false
remove_orphans=false
assume_yes=false
log_services=()
log_since=""
tail_lines=""
wait_timeout=""
env_file_override=""
project_name=""
gpu_device=""
gpu_device_provided=false
autotune_mode=""
autotune_provided=false
force_autotune=false
frontend_flag=false
no_frontend_flag=false

while (($#)); do
    # Accept --option=value as well as --option value, for every option that
    # takes one. --tail=5 in particular was rejected as an unknown option, which
    # is the spelling anyone who has used `docker compose logs` reaches for.
    case "$1" in
        --tail=*|--since=*|--service=*|--wait-timeout=*|--env-file=*|--project-name=*|--gpu-device=*|--autotune=*)
            set -- "${1%%=*}" "${1#*=}" "${@:2}"
            ;;
    esac
    case "$1" in
        --no-build)     no_build=true ;;
        --wait)         wait_for_health=true ;;
        --skip-verify)  skip_verify=true ;;
        --show-secrets) show_secrets=true ;;
        --dry-run)      dry_run=true ;;
        --embedder-only)    embedder_only_flag=true ;;
        --full)             full_flag=true ;;
        --no-mcp)           no_mcp_flag=true ;;
        --mcp)              mcp_flag=true ;;
        --private-embedder) private_embedder_flag=true ;;
        --public-embedder)  public_embedder_flag=true ;;
        --follow)           follow_logs=true ;;
        --volumes)          remove_volumes=true ;;
        --remove-orphans)   remove_orphans=true ;;
        --yes)              assume_yes=true ;;
        --force-autotune)    force_autotune=true ;;
        --frontend)         frontend_flag=true ;;
        --no-frontend)      no_frontend_flag=true ;;
        --autotune)
            if [[ "$autotune_provided" == true ]]; then
                fail_usage "--autotune cannot be repeated."
            fi
            shift
            if (($# == 0)); then
                fail_usage "--autotune requires a mode."
            fi
            autotune_mode="$1"
            autotune_provided=true
            ;;
        --gpu-device)
            shift
            if (($# == 0)); then
                fail_usage "--gpu-device requires a device."
            fi
            if [[ "$1" == *$'\n'* || "$1" == *$'\r'* ]]; then
                fail_usage "GPU device selector must be a single line."
            fi
            gpu_device="$1"
            gpu_device_provided=true
            ;;
        --service)
            shift
            if (($# == 0)); then
                fail_usage "--service requires a service name."
            fi
            # Repeat the option, or comma-separate the names, or both. The
            # comma form exists for parity: a PowerShell parameter cannot be
            # repeated, so -Service takes a list, and an operator translating
            # one command line into the other should not have to notice.
            IFS=',' read -r -a service_names <<< "$1"
            if ((${#service_names[@]} == 0)); then
                fail_usage "--service requires a service name."
            fi
            for service_name in "${service_names[@]}"; do
                if [[ -n "$service_name" ]]; then
                    log_services+=("$service_name")
                fi
            done
            ;;
        --since)
            shift
            if (($# == 0)); then
                fail_usage "--since requires a timestamp or duration."
            fi
            log_since="$1"
            ;;
        --tail)
            shift
            if (($# == 0)); then
                fail_usage "--tail requires a positive line count."
            fi
            tail_lines="$1"
            ;;
        --wait-timeout)
            shift
            if (($# == 0)); then
                fail_usage "--wait-timeout requires a number of seconds."
            fi
            wait_timeout="$1"
            ;;
        --env-file)
            shift
            if (($# == 0)); then
                fail_usage "--env-file requires a path."
            fi
            env_file_override="$1"
            ;;
        --project-name)
            shift
            if (($# == 0)); then
                fail_usage "--project-name requires a name."
            fi
            project_name="$1"
            ;;
        -h|--help)
            command_name="help"
            ;;
        --*)
            fail_usage "Unknown option '$1'."
            ;;
        *)
            positionals+=("$1")
            ;;
    esac
    shift
done

if ((${#positionals[@]} > 2)); then
    fail_usage "Too many positional arguments."
fi

profile="${positionals[0]:-}"
embedder_profile="${positionals[1]:-}"
all_profiles=(--profile cpu --profile gpu)
profile_provided=false
embedder_profile_provided=false

if [[ -n "$profile" ]]; then
    profile_provided=true
    profile="$(lowercase "$profile")"
    if [[ "$profile" == "auto" ]]; then
        case "$command_name" in
            up|config|build|pull) ;;
            *) fail_usage "Hardware auto supports up, config, build and pull." ;;
        esac
    elif [[ "$profile" != "cpu" && "$profile" != "gpu" ]]; then
        fail_usage "Hardware profile must be 'cpu' or 'gpu', got '$profile'."
    fi
fi

if [[ -n "$embedder_profile" ]]; then
    embedder_profile_provided=true
    embedder_profile="$(lowercase "$embedder_profile")"
    if [[ "$embedder_profile" != "basic" && "$embedder_profile" != "advanced" ]]; then
        fail_usage \
            "Smart Embedder profile must be 'basic' or 'advanced', got '$embedder_profile'."
    fi
fi

if [[ -n "$tail_lines" && ! "$tail_lines" =~ ^[1-9][0-9]*$ ]]; then
    fail_usage "--tail requires a positive integer."
fi

if [[ -n "$wait_timeout" && ! "$wait_timeout" =~ ^[1-9][0-9]*$ ]]; then
    fail_usage "--wait-timeout requires a positive integer."
fi

if ! allowed="$(allowed_options_for "$command_name")"; then
    fail_usage "Unknown command '$command_name'."
fi

provided_options=()
if [[ "$autotune_provided" == true ]]; then
    if [[ -z "${autotune_mode//[[:space:]]/}" ||
          "$autotune_mode" == *$'\n'* || "$autotune_mode" == *$'\r'* ]]; then
        fail_usage "--autotune requires one nonempty mode."
    fi
    provided_options+=(autotune)
fi
[[ "$force_autotune" == true ]] && provided_options+=(force-autotune)
[[ "$frontend_flag" == true ]] && provided_options+=(frontend)
[[ "$no_frontend_flag" == true ]] && provided_options+=(no-frontend)
[[ "$gpu_device_provided" == true ]] && provided_options+=(gpu-device)
[[ "$profile_provided" == true ]] && provided_options+=(hardware)
[[ "$embedder_profile_provided" == true ]] && provided_options+=(embedder-profile)
[[ "$no_build" == true ]] && provided_options+=(no-build)
[[ "$wait_for_health" == true ]] && provided_options+=(wait)
[[ "$skip_verify" == true ]] && provided_options+=(skip-verify)
[[ "$show_secrets" == true ]] && provided_options+=(show-secrets)
[[ "$dry_run" == true ]] && provided_options+=(dry-run)
[[ "$embedder_only_flag" == true ]] && provided_options+=(embedder-only)
[[ "$full_flag" == true ]] && provided_options+=(full)
[[ "$no_mcp_flag" == true ]] && provided_options+=(no-mcp)
[[ "$mcp_flag" == true ]] && provided_options+=(mcp)
[[ "$private_embedder_flag" == true ]] && provided_options+=(private-embedder)
[[ "$public_embedder_flag" == true ]] && provided_options+=(public-embedder)
[[ "$follow_logs" == true ]] && provided_options+=(follow)
[[ "$remove_volumes" == true ]] && provided_options+=(volumes)
[[ "$remove_orphans" == true ]] && provided_options+=(remove-orphans)
[[ "$assume_yes" == true ]] && provided_options+=(yes)
((${#log_services[@]})) && provided_options+=(service)
[[ -n "$log_since" ]] && provided_options+=(since)
[[ -n "$tail_lines" ]] && provided_options+=(tail)
[[ -n "$wait_timeout" ]] && provided_options+=(wait-timeout)
[[ -n "$env_file_override" ]] && provided_options+=(env-file)
[[ -n "$project_name" ]] && provided_options+=(project-name)

for option in "${provided_options[@]:-}"; do
    [[ -n "$option" ]] || continue
    case " $allowed " in
        *" $option "*) ;;
        *) fail_usage \
            "Command '$command_name' does not accept $(option_display_name "$option")." ;;
    esac
done

if [[ "$command_name" == "help" ]]; then
    usage
    exit 0
fi

# `version` reports what is installed, so it runs before -- and instead of --
# the checks that refuse to continue when something is missing. Being told
# "Docker Compose: not available" is the answer an operator came for; exiting 1
# with the same information on stderr is not.
if [[ "$command_name" == "version" ]]; then
    echo "Launcher: $LAUNCHER_VERSION (stack.sh)"
    if command -v docker >/dev/null 2>&1; then
        docker_version="$(docker version --format '{{.Server.Version}}' 2>/dev/null)" \
            || docker_version=""
        if [[ -n "$docker_version" ]]; then
            echo "Docker Engine: $docker_version"
        else
            echo "Docker Engine: not reachable (the CLI is installed; the daemon did not answer)"
        fi
        if compose_version="$(docker compose version --short 2>/dev/null)"; then
            echo "Docker Compose: $(printf '%s' "$compose_version" | tr -d '[:space:]')"
        else
            echo "Docker Compose: not available"
        fi
    else
        echo "Docker Engine: not installed"
        echo "Docker Compose: not available"
    fi
    echo "Required: Docker Compose ${MINIMUM_COMPOSE_MAJOR}.${MINIMUM_COMPOSE_MINOR}.${MINIMUM_COMPOSE_PATCH} or newer"
    exit 0
fi

# --- prerequisites ---------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not available on PATH." >&2
    exit 1
fi
if ! compose_version="$(docker compose version --short 2>/dev/null)"; then
    echo "[ERROR] Docker Compose v2 is not available." >&2
    exit 1
fi
compose_version="$(printf '%s' "$compose_version" | tr -d '[:space:]')"
if ! compose_version_supported "$compose_version"; then
    echo "[ERROR] Docker Compose ${MINIMUM_COMPOSE_MAJOR}.${MINIMUM_COMPOSE_MINOR}.${MINIMUM_COMPOSE_PATCH} or newer is required, found '${compose_version}'." >&2
    echo "        This stack uses 'docker compose config --environment', '!reset' in compose.embedder-private.yml, and '!override' in compose.gpu-device.yml." >&2
    exit 1
fi
# `docker compose version` never contacts the daemon, so without this probe the
# first real command failed with Docker's own connect error instead of a message
# an operator can act on.
#
# Only the commands that actually reach the daemon are gated on it. `config` and
# `verify` resolve Compose entirely client-side, and being able to check a
# configuration before starting Docker Desktop is exactly when an operator wants
# them.
#
# --dry-run executes no planned mutation, so it does not need a daemon either.
case "$command_name" in
    up|down|logs|status|build|pull|restart)
        if [[ "$dry_run" == true ]]; then
            :
        elif ! docker info >/dev/null 2>&1; then
            echo "[ERROR] Docker daemon not reachable -- start Docker Desktop / the docker service." >&2
            exit 1
        fi
        ;;
esac

requires_configured_environment=false
case "$command_name" in
    up|config|verify|build|pull|restart|urls)
    # 'verify' answers "is *my* configuration sound", so it deliberately does
    # not fall back to .env.example: verifying the versioned template instead
    # would silently answer a different question, and the same holds for
    # `urls`: it reports where *this* stack is published.
    requires_configured_environment=true
    ;;
esac

if [[ -n "$env_file_override" ]]; then
    # An explicit --env-file is the operator's own choice of configuration, so
    # it is required to exist even for the recovery commands: silently falling
    # back to .env.example would answer a question they did not ask.
    if [[ ! -f "$env_file_override" ]]; then
        echo "[ERROR] Missing environment file $env_file_override." >&2
        exit 1
    fi
    runtime_env_file="$env_file_override"
elif [[ "$requires_configured_environment" == true ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "[ERROR] Missing $ENV_FILE. Copy .env.example to .env and configure it first." >&2
        exit 1
    fi
    runtime_env_file="$ENV_FILE"
else
    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        echo "[ERROR] Missing repository template $ENV_EXAMPLE." >&2
        exit 1
    fi
    # Recovery commands use stable defaults and remain available even when .env
    # is missing or malformed.
    runtime_env_file="$ENV_EXAMPLE"
fi

base=(docker compose)
if [[ -n "$project_name" ]]; then
    base+=(--project-name "$project_name")
fi
base+=(--file "$COMPOSE_FILE" --env-file "$runtime_env_file")

# --- recovery commands: pure shell, no planner, no Python ------------------

case "$command_name" in
    down)
        down_arguments=("${base[@]}" "${all_profiles[@]}" down)
        if [[ "$remove_volumes" == true ]]; then
            down_arguments+=(--volumes)
        fi
        if [[ "$remove_orphans" == true ]]; then
            down_arguments+=(--remove-orphans)
        fi
        if [[ "$dry_run" == true ]]; then
            print_dry_run_step "stop the stack" "${down_arguments[@]}"
            exit 0
        fi
        # --volumes destroys the Hugging Face model cache, which is tens of
        # gigabytes and takes a long download to rebuild. It is confirmed
        # interactively, and refused outright when there is nobody to ask.
        if [[ "$remove_volumes" == true && "$assume_yes" == false ]]; then
            if [[ -t 0 ]]; then
                echo "This removes every named volume of the stack, including the Hugging Face model cache." >&2
                echo "Rebuilding it means downloading every model again." >&2
                printf 'Type yes to continue: ' >&2
                read -r confirmation || confirmation=""
                if [[ "$(lowercase "$confirmation")" != "yes" ]]; then
                    echo "[ERROR] Aborted; nothing was removed." >&2
                    exit 2
                fi
            else
                echo "[ERROR] --volumes destroys every named volume, including the Hugging Face model cache." >&2
                echo "        Standard input is not a terminal, so it cannot be confirmed here; re-run with --yes." >&2
                exit 2
            fi
        fi
        "${down_arguments[@]}"
        exit $?
        ;;
    logs)
        compose_arguments=("${base[@]}")
        if [[ "$profile_provided" == true ]]; then
            compose_arguments+=(--profile "$profile" logs)
        else
            compose_arguments+=("${all_profiles[@]}" logs)
        fi
        if [[ -n "$tail_lines" ]]; then
            compose_arguments+=(--tail "$tail_lines")
        fi
        if [[ -n "$log_since" ]]; then
            compose_arguments+=(--since "$log_since")
        fi
        # --follow and --tail are independent in Compose; they were mutually
        # exclusive here only because --follow was the implicit default.
        # Following stays the default when neither is asked for.
        if [[ "$follow_logs" == true || ( -z "$tail_lines" && -z "$log_since" ) ]]; then
            compose_arguments+=(--follow)
        fi
        if ((${#log_services[@]})); then
            compose_arguments+=("${log_services[@]}")
        elif [[ "$profile_provided" == true ]]; then
            compose_arguments+=(middleware "smart-embedder-$profile" smart-qdrant-mcp)
        fi
        if [[ "$dry_run" == true ]]; then
            print_dry_run_step "show container logs" "${compose_arguments[@]}"
            exit 0
        fi
        "${compose_arguments[@]}"
        exit $?
        ;;
    status)
        compose_arguments=("${base[@]}")
        if [[ "$profile_provided" == true ]]; then
            compose_arguments+=(
                --profile "$profile" ps middleware "smart-embedder-$profile" smart-qdrant-mcp
            )
        else
            compose_arguments+=("${all_profiles[@]}" ps)
        fi
        if [[ "$dry_run" == true ]]; then
            print_dry_run_step "show container status" "${compose_arguments[@]}"
            exit 0
        fi
        "${compose_arguments[@]}"
        exit $?
        ;;
esac

# --- planned commands: up, config, verify ----------------------------------

preflight_python=""
if ! preflight_python="$(find_python)"; then
    echo "[ERROR] No Python interpreter found; '$command_name' needs one." >&2
    exit 2
fi

if [[ "$embedder_profile_provided" == true ]]; then
    export SMART_EMBEDDER_PROFILE="$embedder_profile"
fi

compose_environment="$("${base[@]}" config --environment)"

resolved_value() {
    local key="$1"
    local default_value="${2:-}"
    local line
    local found=false
    local value=""

    while IFS= read -r line; do
        if [[ "$line" == "$key="* ]]; then
            value="${line#*=}"
            found=true
        fi
    done <<< "$compose_environment"

    if [[ "$found" == true ]]; then
        printf '%s' "$value"
    else
        printf '%s' "$default_value"
    fi
}

# The planner receives the resolved environment and the parsed CLI intent on
# stdin, never on argv: the environment carries bearer tokens and arguments are
# world-readable via ps.
planner_input="$compose_environment"$'\n'
planner_input+="--cli-command=${command_name}"$'\n'
if [[ "$autotune_provided" == true ]]; then
    planner_input+="--cli-autotune=${autotune_mode}"$'\n'
fi
planner_input+="--cli-force-autotune=${force_autotune}"$'\n'
planner_input+="--cli-flag-frontend=${frontend_flag}"$'\n'
planner_input+="--cli-flag-no-frontend=${no_frontend_flag}"$'\n'
if [[ "$profile_provided" == true ]]; then
    planner_input+="--cli-hardware=${profile}"$'\n'
fi
if [[ "$embedder_profile_provided" == true ]]; then
    planner_input+="--cli-embedder-profile=${embedder_profile}"$'\n'
fi
planner_input+="--cli-no-build=${no_build}"$'\n'
planner_input+="--cli-wait=${wait_for_health}"$'\n'
planner_input+="--cli-wait-timeout=${wait_timeout}"$'\n'
planner_input+="--cli-project-name=${project_name}"$'\n'
# The opposed flags are passed through raw. Folding them into one value, and
# refusing a pair that contradicts itself, is the planner's job -- doing it here
# would put the same decision in two languages again.
planner_input+="--cli-flag-embedder-only=${embedder_only_flag}"$'\n'
planner_input+="--cli-flag-full=${full_flag}"$'\n'
planner_input+="--cli-flag-mcp=${mcp_flag}"$'\n'
planner_input+="--cli-flag-no-mcp=${no_mcp_flag}"$'\n'
planner_input+="--cli-flag-public-embedder=${public_embedder_flag}"$'\n'
planner_input+="--cli-flag-private-embedder=${private_embedder_flag}"$'\n'
planner_input+="--cli-compose-file=${COMPOSE_FILE}"$'\n'
planner_input+="--cli-env-file=${runtime_env_file}"$'\n'
planner_input+="--cli-embedder-private-file=${EMBEDDER_PRIVATE_COMPOSE_FILE}"$'\n'
planner_input+="--cli-gpu-overlay-file=${GPU_DEVICE_COMPOSE_FILE}"$'\n'
if [[ "$gpu_device_provided" == true ]]; then
    planner_input+="--cli-gpu-device=${gpu_device}"$'\n'
fi
if ! gpu_facts="$(printf '%s' "$planner_input" \
    | "$preflight_python" "${SCRIPT_DIR}/scripts/stack_gpu_probe.py")"; then
    exit 2
fi
if [[ -n "$gpu_facts" ]]; then
    planner_input+="${gpu_facts}"$'\n'
fi

if ! plan="$(printf '%s' "$planner_input" \
    | "$preflight_python" "${SCRIPT_DIR}/scripts/stack_core.py" --format=sh)"; then
    exit 2
fi

# Read the plan into the shell. A here-string rather than a pipe, so the loop
# runs in this shell and its exports and variables survive it.
plan_steps=()
plan_descriptions=()
plan_messages=()
plan_summary=()
warn_if_gpu=false
resolved_embedder_only=false
resolved_mcp_server_enabled=true
current_step=""
while IFS= read -r plan_line; do
    # Defensive: the planner writes LF explicitly, but a stray CR here would
    # otherwise become part of the first argument of a step.
    plan_line="${plan_line%$'\r'}"
    plan_kind="${plan_line%% *}"
    plan_rest="${plan_line#* }"
    case "$plan_kind" in
        env)
            export "${plan_rest?}"
            ;;
        resolved)
            case "$plan_rest" in
                embedder_only=*) resolved_embedder_only="${plan_rest#*=}" ;;
                mcp_server_enabled=*) resolved_mcp_server_enabled="${plan_rest#*=}" ;;
            esac
            ;;
        warn_if_gpu)
            warn_if_gpu="$plan_rest"
            ;;
        message)
            plan_messages+=("$plan_rest")
            ;;
        warning)
            echo "[WARN] $plan_rest" >&2
            ;;
        summary)
            plan_summary+=("$plan_rest")
            ;;
        step)
            if [[ -n "$current_step" ]]; then
                plan_steps+=("$current_step")
            fi
            current_step=""
            plan_descriptions+=("$plan_rest")
            ;;
        arg)
            # Steps are stored as newline-joined argv and split back with a
            # newline IFS at execution time, because bash 3.2 has no nested
            # arrays.
            if [[ -z "$current_step" ]]; then
                current_step="$plan_rest"
            else
                current_step+=$'\n'"$plan_rest"
            fi
            ;;
    esac
done <<< "$plan"
if [[ -n "$current_step" ]]; then
    plan_steps+=("$current_step")
fi

# Both verifiers signal a violation with exit 1. The launcher's own contract
# reserves 1 for an unmet prerequisite and 2 for a failed preflight, so every
# non-zero result from either one is mapped to 2 here. Their own contracts are
# unchanged; only the caller reinterprets them.
run_verification() {
    warn_insecure_env_permissions

    # Every token invariant that spans services lives in one place:
    # scripts/verify_token_config.py. Values go over stdin, not arguments,
    # because they are bearer tokens and arguments are world-readable via ps.
    local preflight_input=""
    local name
    # The last three names were removed by the multi-token change; they are
    # passed on purpose so the preflight can refuse a stale .env.
    # MCP_AUTH_MODE tells the preflight whether the MCP server's own empty-list
    # boot check applies: under oauth the token list is ignored.
    for name in \
        MIDDLEWARE_API_TOKENS SMART_EMBEDDER_API_TOKENS MCP_API_TOKENS \
        MIDDLEWARE_CLIENT_TOKEN SMART_EMBEDDER_CLIENT_TOKEN \
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL MIDDLEWARE_BIND_HOST \
        SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL \
        MCP_ALLOW_UNAUTHENTICATED_LOCAL MCP_BIND_HOST \
        MCP_AUTH_MODE \
        MIDDLEWARE_API_TOKEN SMART_EMBEDDER_API_TOKEN ALLOW_UNAUTHENTICATED_LOCAL
    do
        preflight_input+="${name}=$(resolved_value "$name")"$'\n'
    done
    preflight_input+="MCP_SERVER_ENABLED=${resolved_mcp_server_enabled}"$'\n'
    preflight_input+="EMBEDDER_ONLY=${resolved_embedder_only}"$'\n'
    if ! printf '%s' "$preflight_input" \
        | "$preflight_python" "${SCRIPT_DIR}/scripts/verify_token_config.py"; then
        exit 2
    fi

    # The compose hardening invariants used to run against .env.example only, by
    # hand. Running them against the operator's own env file is the point.
    if ! "$preflight_python" "${SCRIPT_DIR}/scripts/verify_compose_security.py" \
        --env-file "$runtime_env_file"; then
        exit 2
    fi
}

# --dry-run skips verifiers, so it reports what verification would do instead of
# doing it. That is what makes "--skip-verify really skips it" checkable without
# starting anything.
verification_would_run=false
if [[ "$command_name" == "verify" ]]; then
    verification_would_run=true
elif [[ "$command_name" == "up" && "$skip_verify" == false ]]; then
    verification_would_run=true
fi

if [[ "$dry_run" == true ]]; then
    # Only the two commands verification applies to say anything about it.
    # Reporting "verification would be skipped" under `urls` would describe a
    # decision nobody made.
    if [[ "$command_name" == "up" || "$command_name" == "verify" ]]; then
        if [[ "$verification_would_run" == true ]]; then
            echo "[dry-run] configuration verification would run." >&2
        else
            echo "[dry-run] configuration verification would be skipped." >&2
        fi
    fi
else
    if [[ "$verification_would_run" == true ]]; then
        run_verification
    fi
    if [[ "$command_name" == "verify" ]]; then
        exit 0
    fi
fi

# Manual GPU startup keeps the advisory probe, skipped under --dry-run.
# Required auto/selected-GPU discovery already completed before planning.
if [[ "$warn_if_gpu" == true && "$dry_run" == false ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] nvidia-smi is unavailable locally; Docker Compose will perform the authoritative GPU runtime check." >&2
    elif ! nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] Local nvidia-smi failed; continuing because the active Docker context may be remote." >&2
    fi
fi

for plan_message in "${plan_messages[@]:-}"; do
    [[ -n "$plan_message" ]] && echo "$plan_message" >&2
done

plan_step_index=0
for step in "${plan_steps[@]:-}"; do
    [[ -n "$step" ]] || continue
    step_argv=()
    while IFS= read -r step_argument; do
        step_argv+=("$step_argument")
    done <<< "$step"
    if [[ "$dry_run" == true ]]; then
        print_dry_run_step "${plan_descriptions[$plan_step_index]}" "${step_argv[@]}"
        plan_step_index=$((plan_step_index + 1))
        continue
    fi
    if [[ "$command_name" == "config" && "$show_secrets" == false ]]; then
        "${step_argv[@]}" | redact_compose_secrets
    else
        if [[ "$command_name" == "config" ]]; then
            echo "[WARN] --show-secrets: every bearer token in the stack is printed in cleartext below." >&2
        fi
        "${step_argv[@]}"
    fi
done

# The summary is primary output, so it goes to stdout, unlike the messages and
# warnings above. It is printed after every step has succeeded -- the loop above
# aborts the script on the first failure -- which is what makes it a report of
# what is now true rather than of what was intended. `urls` has no steps at all
# and is nothing but this.
for summary_line in "${plan_summary[@]:-}"; do
    if [[ -n "$summary_line" ]]; then
        echo "$summary_line"
    fi
done

# Explicit, because the exit status of this script is the status of its last
# command: with an empty summary the loop above ends on a false test, and the
# launcher reported failure for a run in which everything had succeeded.
exit 0
