#!/bin/bash

# This script automates the download, configuration, and execution of SonarQube Scanner on Unix/Linux systems.
# It also includes validation of configuration parameters, connectivity tests to the SonarQube server,
# and checks the Quality Gate status after the scan.

# Usage:
#   ./sonarqube-unix.sh [OPTIONS]
#
# Options:
#   --sonar-scanner-version <version>  SonarQube Scanner version (default: "7.2.0.5079")
#   --project-key <key>                SonarQube project key (default: "PROJECT_KEY")
#   --sonar-host-url <url>             SonarQube server URL (default: "https://ihsonarqube.ihcantabria.com")
#   --sonar-token <token>              SonarQube authentication token (default: "SONAR_TOKEN_KEY")
#   --project-dir <path>               Project directory to scan (default: ".")
#   --skip-connectivity-test           Skip server connectivity test
#   --skip-quality-gate-check          Skip Quality Gate verification
#   --help                             Show this help message

set -e  # Exit on error

# Default configuration
SONAR_SCANNER_VERSION="7.2.0.5079"
PROJECT_KEY="IHCantabria_template.python.lib_3a6e6e1c-9615-45f0-b54f-db978ffc9844"
SONAR_HOST_URL="https://ihsonarqube.ihcantabria.com"
SONAR_TOKEN="${SONAR_TOKEN:-}"  # Use environment variable or pass via --sonar-token
PROJECT_DIR="."
SKIP_CONNECTIVITY_TEST=false
SKIP_QUALITY_GATE_CHECK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sonar-scanner-version)
            SONAR_SCANNER_VERSION="$2"
            shift 2
            ;;
        --project-key)
            PROJECT_KEY="$2"
            shift 2
            ;;
        --sonar-host-url)
            SONAR_HOST_URL="$2"
            shift 2
            ;;
        --sonar-token)
            SONAR_TOKEN="$2"
            shift 2
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --skip-connectivity-test)
            SKIP_CONNECTIVITY_TEST=true
            shift
            ;;
        --skip-quality-gate-check)
            SKIP_QUALITY_GATE_CHECK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sonar-scanner-version <version>  SonarQube Scanner version (default: 7.2.0.5079)"
            echo "  --project-key <key>                SonarQube project key (default: PROJECT_KEY)"
            echo "  --sonar-host-url <url>             SonarQube server URL (default: https://ihsonarqube.ihcantabria.com)"
            echo "  --sonar-token <token>              SonarQube authentication token (default: SONAR_TOKEN_KEY)"
            echo "  --project-dir <path>               Project directory to scan (default: .)"
            echo "  --skip-connectivity-test           Skip server connectivity test"
            echo "  --skip-quality-gate-check          Skip Quality Gate verification"
            echo "  --help                             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configuration
WORKSPACE_ROOT=$(pwd)
TEMP_DOWNLOAD_DIR="$WORKSPACE_ROOT/sonar-temp-download"
SONAR_DIRECTORY="$WORKSPACE_ROOT/.sonar"
SCANNER_WORK_DIR="$WORKSPACE_ROOT/.scannerwork"

# Detect OS and architecture
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case "$os" in
        linux*)
            OS_TYPE="linux"
            ;;
        darwin*)
            OS_TYPE="macosx"
            ;;
        *)
            echo "Error: Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    case "$arch" in
        x86_64|amd64)
            ARCH_TYPE="x64"
            ;;
        aarch64|arm64)
            ARCH_TYPE="aarch64"
            ;;
        *)
            echo "Error: Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    echo "Detected platform: $OS_TYPE-$ARCH_TYPE"
}

# Set scanner paths after platform detection
set_scanner_paths() {
    SONAR_SCANNER_HOME="$SONAR_DIRECTORY/sonar-scanner-$SONAR_SCANNER_VERSION-$OS_TYPE-$ARCH_TYPE"
    SCANNER_ZIP="$TEMP_DOWNLOAD_DIR/sonar-scanner.zip"
    SCANNER_EXECUTABLE="$SONAR_SCANNER_HOME/bin/sonar-scanner"
}

# Utility: Clean partial files and working directories
cleanup_partial_files() {
    local include_scanner=$1
    
    echo "Cleaning up partial files..."
    
    rm -rf "$TEMP_DOWNLOAD_DIR" 2>/dev/null || true
    rm -rf "$SCANNER_WORK_DIR" 2>/dev/null || true
    
    if [ "$include_scanner" = "true" ]; then
        rm -rf "$SONAR_DIRECTORY" 2>/dev/null || true
    fi
}

# Utility: Download SonarQube Scanner
download_sonar_scanner() {
    if [ -f "$SCANNER_EXECUTABLE" ] && [ -x "$SCANNER_EXECUTABLE" ]; then
        echo "SonarQube Scanner already exists and is functional. Reusing."
        return
    fi
    
    echo "Downloading SonarQube Scanner v$SONAR_SCANNER_VERSION..."
    
    # Create directories
    mkdir -p "$TEMP_DOWNLOAD_DIR"
    mkdir -p "$SONAR_DIRECTORY"
    
    local scanner_url="https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SONAR_SCANNER_VERSION-$OS_TYPE-$ARCH_TYPE.zip"
    
    echo "Downloading from: $scanner_url"
    
    if command -v curl &> /dev/null; then
        curl -L -o "$SCANNER_ZIP" "$scanner_url" || {
            echo "Error: Failed to download SonarQube Scanner"
            cleanup_partial_files true
            exit 1
        }
    elif command -v wget &> /dev/null; then
        wget -O "$SCANNER_ZIP" "$scanner_url" || {
            echo "Error: Failed to download SonarQube Scanner"
            cleanup_partial_files true
            exit 1
        }
    else
        echo "Error: Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    # Verify download
    if [ ! -f "$SCANNER_ZIP" ] || [ ! -s "$SCANNER_ZIP" ]; then
        echo "Error: Failed to download SonarQube Scanner or file is empty"
        cleanup_partial_files true
        exit 1
    fi
    
    echo "Extracting SonarQube Scanner..."
    
    if command -v unzip &> /dev/null; then
        unzip -q "$SCANNER_ZIP" -d "$SONAR_DIRECTORY" || {
            echo "Error: Failed to extract SonarQube Scanner"
            cleanup_partial_files true
            exit 1
        }
    else
        echo "Error: unzip command not found. Please install unzip."
        cleanup_partial_files true
        exit 1
    fi
    
    # Verify extraction
    if [ ! -f "$SCANNER_EXECUTABLE" ]; then
        echo "Error: SonarQube Scanner executable not found after extraction"
        cleanup_partial_files true
        exit 1
    fi
    
    # Make executable
    chmod +x "$SCANNER_EXECUTABLE"
    
    # Cleanup download files
    rm -rf "$TEMP_DOWNLOAD_DIR"
    
    echo "SonarQube Scanner setup completed successfully."
}

# Utility: Validate configuration
validate_configuration() {
    echo "Validating configuration..."
    local has_errors=false
    
    # Validate SonarQube token
    if [ -z "$SONAR_TOKEN" ] || [ "$SONAR_TOKEN" = "SONAR_TOKEN_KEY" ]; then
        echo "Error: Invalid SONAR_TOKEN. Please provide a valid SonarQube authentication token."
        has_errors=true
    fi
    
    # Validate project key
    if [ -z "$PROJECT_KEY" ] || [ "$PROJECT_KEY" = "PROJECT_KEY" ]; then
        echo "Error: Invalid PROJECT_KEY. Please provide a valid SonarQube project key."
        has_errors=true
    fi
    
    # Validate SonarQube host URL
    if [ -z "$SONAR_HOST_URL" ]; then
        echo "Error: Invalid SONAR_HOST_URL. Please provide a valid SonarQube server URL."
        has_errors=true
    fi
    
    # Validate project directory
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Error: Project directory does not exist: $PROJECT_DIR"
        has_errors=true
    fi
    
    # Validate SonarQube Scanner version format
    if ! [[ "$SONAR_SCANNER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Warning: SonarQube Scanner version format may be incorrect: $SONAR_SCANNER_VERSION"
    fi
    
    if [ "$has_errors" = true ]; then
        echo "Error: Configuration validation failed. Please fix the errors above."
        exit 1
    fi
    
    echo "Configuration validation passed successfully."
}

# Utility: Test SonarQube server connectivity
test_sonarqube_connectivity() {
    if [ "$SKIP_CONNECTIVITY_TEST" = true ]; then
        echo "Connectivity test skipped by user request."
        return
    fi
    
    echo "Testing SonarQube server connectivity..."
    
    local status_code
    if command -v curl &> /dev/null; then
        status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$SONAR_HOST_URL/api/system/status" 2>/dev/null || echo "000")
    elif command -v wget &> /dev/null; then
        status_code=$(wget --spider --timeout=10 -S "$SONAR_HOST_URL/api/system/status" 2>&1 | grep "HTTP/" | awk '{print $2}' || echo "000")
    else
        echo "Warning: Neither curl nor wget available. Skipping connectivity test."
        return
    fi
    
    if [ "$status_code" = "200" ]; then
        echo "SonarQube server is accessible."
        return 0
    else
        echo "Warning: SonarQube server responded with status: $status_code"
        echo "Proceeding with scan (server may still be accessible during scan)..."
        return 1
    fi
}

# Utility: Configure SonarQube environment
configure_sonar_environment() {
    echo "Configuring SonarQube environment..."
    
    # Set environment variables
    export SONAR_SCANNER_HOME="$SONAR_SCANNER_HOME"
    export SONAR_SCANNER_OPTS="-server"
    export SONAR_TOKEN="$SONAR_TOKEN"
    
    # Add to PATH if not already present
    local bin_path="$SONAR_SCANNER_HOME/bin"
    if [[ ":$PATH:" != *":$bin_path:"* ]]; then
        export PATH="$bin_path:$PATH"
        echo "Added SonarQube Scanner to PATH: $bin_path"
    else
        echo "SonarQube Scanner already in PATH"
    fi
}

# Utility: Run SonarQube scan
run_sonar_scan() {
    echo "Verifying project directory..."
    local absolute_project_dir=$(realpath "$PROJECT_DIR" 2>/dev/null || readlink -f "$PROJECT_DIR" 2>/dev/null)
    
    if [ -z "$absolute_project_dir" ] || [ ! -d "$absolute_project_dir" ]; then
        echo "Error: Project directory does not exist: $PROJECT_DIR"
        exit 1
    fi
    
    echo "Project directory: $absolute_project_dir"
    echo "Starting SonarQube scan..."
    
    # Set working directory
    local original_location=$(pwd)
    cd "$absolute_project_dir"
    
    echo "Executing: $SCANNER_EXECUTABLE"
    
    "$SCANNER_EXECUTABLE" \
        -Dsonar.projectKey="$PROJECT_KEY" \
        -Dsonar.sources=. \
        -Dsonar.host.url="$SONAR_HOST_URL" \
        -Dsonar.token="$SONAR_TOKEN" \
        -Dsonar.working.directory="$SCANNER_WORK_DIR" || {
        echo "Error: SonarQube scan failed"
        cd "$original_location"
        exit 1
    }
    
    echo "SonarQube scan completed successfully."
    
    # Restore original location
    cd "$original_location"
    
    # Clean up scanner work directory
    if [ -d "$SCANNER_WORK_DIR" ]; then
        echo "Cleaning up scanner work directory..."
        rm -rf "$SCANNER_WORK_DIR"
    fi
}

# Utility: Validate SonarQube token and permissions
test_sonarqube_authentication() {
    echo "Validating SonarQube authentication..."
    
    local auth_test_url="$SONAR_HOST_URL/api/authentication/validate"
    local response
    
    if command -v curl &> /dev/null; then
        response=$(curl -s -H "Authorization: Bearer $SONAR_TOKEN" --max-time 10 "$auth_test_url" 2>/dev/null || echo '{"valid":false}')
    elif command -v wget &> /dev/null; then
        response=$(wget -q --timeout=10 --header="Authorization: Bearer $SONAR_TOKEN" -O - "$auth_test_url" 2>/dev/null || echo '{"valid":false}')
    else
        echo "Warning: Cannot validate authentication without curl or wget."
        return 1
    fi
    
    if command -v jq &> /dev/null; then
        local is_valid=$(echo "$response" | jq -r '.valid' 2>/dev/null || echo "false")
        if [ "$is_valid" = "true" ]; then
            echo "Token authentication successful."
            return 0
        else
            echo "Error: Token is not valid."
            return 1
        fi
    else
        echo "Warning: jq not found. Cannot parse authentication response."
        echo "Install jq for better authentication validation."
        return 1
    fi
}

# Utility: Check Quality Gate status
check_quality_gate() {
    if [ "$SKIP_QUALITY_GATE_CHECK" = true ]; then
        echo "Quality Gate check skipped by user request."
        return
    fi
    
    echo "Checking Quality Gate status..."
    
    # First validate authentication
    if ! test_sonarqube_authentication; then
        echo "Error: Cannot proceed with Quality Gate check due to authentication issues."
        exit 1
    fi
    
    local max_attempts=5
    local attempt=0
    local quality_gate_checked=false
    
    while [ $attempt -lt $max_attempts ] && [ "$quality_gate_checked" = false ]; do
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts - Checking Quality Gate status..."
        
        local quality_gate_url="$SONAR_HOST_URL/api/qualitygates/project_status?projectKey=$PROJECT_KEY"
        local response
        local status_code
        
        if command -v curl &> /dev/null; then
            response=$(curl -s -H "Authorization: Bearer $SONAR_TOKEN" --max-time 15 -w "\n%{http_code}" "$quality_gate_url" 2>/dev/null || echo -e "\n000")
            status_code=$(echo "$response" | tail -n1)
            response=$(echo "$response" | sed '$d')
        elif command -v wget &> /dev/null; then
            response=$(wget -q --timeout=15 --header="Authorization: Bearer $SONAR_TOKEN" -O - "$quality_gate_url" 2>/dev/null || echo "")
            status_code="200"
            if [ -z "$response" ]; then
                status_code="000"
            fi
        else
            echo "Error: Neither curl nor wget available for Quality Gate check."
            return
        fi
        
        if [ "$status_code" = "200" ] && [ -n "$response" ]; then
            if command -v jq &> /dev/null; then
                local project_status=$(echo "$response" | jq -r '.projectStatus.status' 2>/dev/null || echo "")
                
                if [ -n "$project_status" ] && [ "$project_status" != "null" ]; then
                    echo "Quality Gate Status: $project_status"
                    quality_gate_checked=true
                    
                    if [ "$project_status" = "ERROR" ]; then
                        echo -e "\033[0;31mQuality Gate FAILED! The project has critical issues that need to be addressed.\033[0m"
                        
                        # Show detailed information about failed conditions
                        local conditions=$(echo "$response" | jq -r '.projectStatus.conditions[]? | select(.status == "ERROR") | "  - \(.metricKey): \(.actualValue) (threshold: \(.errorThreshold))"' 2>/dev/null)
                        if [ -n "$conditions" ]; then
                            echo -e "\033[0;33mFailed conditions:\033[0m"
                            echo "$conditions"
                        fi
                        
                        echo -e "\033[0;33mPlease check the SonarQube dashboard for detailed analysis: $SONAR_HOST_URL/dashboard?id=$PROJECT_KEY\033[0m"
                        exit 1
                    elif [ "$project_status" = "WARN" ]; then
                        echo "Warning: Quality Gate passed with warnings. Consider reviewing the issues found."
                        
                        # Show warning conditions
                        local warnings=$(echo "$response" | jq -r '.projectStatus.conditions[]? | select(.status == "WARN") | "  - \(.metricKey): \(.actualValue) (threshold: \(.warningThreshold))"' 2>/dev/null)
                        if [ -n "$warnings" ]; then
                            echo -e "\033[0;33mWarning conditions:\033[0m"
                            echo "$warnings"
                        fi
                    else
                        echo "Quality Gate PASSED! No critical issues found."
                    fi
                    break
                else
                    echo "Warning: Quality Gate data not yet available. Analysis may still be processing..."
                    if [ $attempt -lt $max_attempts ]; then
                        sleep 10
                    fi
                fi
            else
                echo "Warning: jq not found. Cannot parse Quality Gate response."
                echo "Install jq for automatic Quality Gate checking."
                break
            fi
        elif [ "$status_code" = "403" ]; then
            echo "Error: Access denied (403) when checking Quality Gate. Possible causes:"
            echo "1. Token lacks permissions for project '$PROJECT_KEY'"
            echo "2. Project key '$PROJECT_KEY' does not exist"
            echo "3. Token does not have 'Browse' permissions on the project"
            exit 1
        elif [ "$status_code" = "404" ]; then
            echo "Warning: Project not found (404). Analysis may not be complete yet..."
            if [ $attempt -lt $max_attempts ]; then
                sleep 10
            fi
        else
            echo "Warning: Could not check Quality Gate status (attempt $attempt)"
            if [ $attempt -lt $max_attempts ]; then
                sleep 10
            fi
        fi
    done
    
    if [ "$quality_gate_checked" = false ]; then
        echo "Warning: Could not retrieve Quality Gate status after $max_attempts attempts."
        echo "Analysis completed but Quality Gate status is unknown."
        echo "Please check manually: $SONAR_HOST_URL/dashboard?id=$PROJECT_KEY"
    fi
}

# Main execution flow
main() {
    echo "=== SonarQube Scanner Automation Script ==="
    echo "Scanner Version: $SONAR_SCANNER_VERSION"
    echo "Project Key: $PROJECT_KEY"
    echo "Host URL: $SONAR_HOST_URL"
    echo "Project Directory: $PROJECT_DIR"
    echo "Skip Connectivity Test: $SKIP_CONNECTIVITY_TEST"
    echo "Skip Quality Gate Check: $SKIP_QUALITY_GATE_CHECK"
    echo "================================================"
    
    detect_platform
    set_scanner_paths
    validate_configuration
    test_sonarqube_connectivity
    download_sonar_scanner
    configure_sonar_environment
    run_sonar_scan
    check_quality_gate
    
    echo "=== SonarQube analysis completed successfully! ==="
}

# Trap errors and cleanup
trap 'cleanup_partial_files false' ERR

# Run main function
main

exit 0
