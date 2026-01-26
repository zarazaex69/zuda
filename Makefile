.PHONY: help build install clean test lint deps tidy bench

BINARY_NAME=zuda
VERSION?=0.1.0
BUILD_DIR=zig-out/bin
INSTALL_DIR=/usr/local/bin

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the CLI binary
	@echo "Building $(BINARY_NAME)..."
	@zig build -Doptimize=ReleaseSafe

install: build ## Install the CLI to system
	@echo "Installing $(BINARY_NAME) to $(INSTALL_DIR)..."
	@sudo cp $(BUILD_DIR)/$(BINARY_NAME) $(INSTALL_DIR)/
	@echo "Installation complete. Run '$(BINARY_NAME)' to get started."

clean: ## Clean build artifacts
	@echo "Cleaning..."
	@rm -rf zig-out zig-cache

test: ## Run tests
	@echo "Running tests..."
	@zig build test

lint: ## Run linters
	@echo "Running zig fmt..."
	@zig fmt src/

deps: ## Download dependencies
	@echo "Fetching dependencies..."
	@zig build --fetch

tidy: ## Clean and rebuild
	@echo "Cleaning and rebuilding..."
	@rm -rf zig-cache
	@zig build

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@zig build -Doptimize=ReleaseFast
	@time ./$(BUILD_DIR)/$(BINARY_NAME)
